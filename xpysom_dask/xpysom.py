from math import sqrt, ceil
from collections import defaultdict
from functools import partial
from warnings import warn
from collections import defaultdict, Counter
from typing import Callable, Iterable, Sequence, Literal
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os

import numpy as np

try:
    # Cupy needs to be imported first.
    # Cudf is crashing containers if it goes first.
    import cupy as cp
    import cudf
    import dask_cudf as dcudf

    default_xp = cp
    GPU_SUPPORTED = True
except ModuleNotFoundError:
    print("WARNING: CuPy could not be imported")
    default_xp = np
    GPU_SUPPORTED = False

try:
    import dask
    import dask.array as da
    import dask.delayed as dd
    import dask.dataframe as ddf

    default_da = True
except ModuleNotFoundError:
    print("WARNING: Dask Arrays could not be imported")
    default_da = False
    
try:
    from dask_ml.decomposition import PCA
except ModuleNotFoundError:
    from sklearn.decomposition import PCA


from .distances import DistanceFunction, euclidean_distance
from .neighborhoods import Neighborhoods
from .utils import find_cpu_cores, find_max_cuda_threads
from .decays import linear_decay, asymptotic_decay, exponential_decay

# In my machine it looks like these are the best performance/memory trade-off.
# As a rule of thumb, executing more items at a time does not decrease
# performance but it may increase the memory footprint without providing
# significant gains.
DEFAULT_CPU_CORE_OVERSUBSCRIPTION = 500

beginning = None
sec_left = None


def print_progress(t, T):
    digits = len(str(T))

    global beginning, sec_left

    if t == -1:
        progress = "\r [ {s:{d}} / {T} ] {s:3.0f}% - ? it/s"
        progress = progress.format(T=T, d=digits, s=0)
        stdout.write(progress)
        beginning = time()
    else:
        sec_left = ((T - t + 1) * (time() - beginning)) / (t + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        sec_elapsed = time() - beginning
        time_elapsed = str(timedelta(seconds=sec_elapsed))[:7]
        progress = "\r [ {t:{d}} / {T} ]".format(t=t + 1, d=digits, T=T)
        progress += " {p:3.0f}%".format(p=100 * (t + 1) / T)
        progress += " - {time_elapsed} elapsed ".format(time_elapsed=time_elapsed)
        progress += " - {time_left} left ".format(time_left=time_left)
        stdout.write(progress)


class XPySom:
    def __init__(
        self,
        x,
        y,
        input_len,
        sigma=0,
        sigmaN=1,
        learning_rate=0.5,
        learning_rateN=0.01,
        decay_function="exponential",
        neighborhood_function="gaussian",
        std_coeff=0.5,
        topology="rectangular",
        inner_dist_type: str | Sequence[str] = "grid",
        PBC: bool = False,
        activation_distance="euclidean",
        activation_distance_kwargs={},
        init: Literal["random", "pca"] = "random",
        random_seed=None,
        n_parallel=0,
        compact_support=False,
        xp=default_xp,
        use_dask=False,
        dask_chunks="auto",
    ):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=min(x,y)/2)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.

        sigmaN : float, optional (default=0.01)
            Spread of the neighborhood function at last iteration.

        learning_rate : float, optional (default=0.5)
            initial learning rate.

        learning_rateN : float, optional (default=0.01)
            final learning rate

        decay_function : string, optional (default='exponential')
            Function that reduces learning_rate and sigma at each iteration.
            Possible values: 'exponential', 'linear', 'aymptotic'

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'norm_p'

        activation_distance_kwargs : dict, optional (default={})
            Pass additional argumets to distance function.
            norm_p:
                p: exponent of the norm-p distance

        random_seed : int, optional (default=None)
            Random seed to use.

        n_parallel : uint, optionam (default=#max_CUDA_threads or 500*#CPUcores)
            Number of samples to be processed at a time. Setting a too low
            value may drastically lower performance due to under-utilization,
            setting a too high value increases memory usage without granting
            any significant performance benefit.

        xp : numpy or cupy, optional (default=cupy if can be imported else numpy)
            Use numpy (CPU) or cupy (GPU) for computations.

        std_coeff: float, optional (default=0.5)
            Used to calculate gausssian exponent denominator:
            d = 2*std_coeff**2*sigma**2

        compact_support: bool, optional (default=False)
            Cut the neighbor function to 0 beyond neighbor radius sigma

        use_dask: bool, optional (default=False)
            Use a distributed SOM based on Dask clustering

        dask_chunks: tuple, optional (default='auto')
            The size of the data chunks that it will be splited up

        """

        if sigma >= x or sigma >= y:
            warn("Warning: sigma is too high for the dimension of the map.")

        self._random_generator = np.random.default_rng(random_seed)

        self.xp = xp

        # Use dask for clustering SOM
        self.use_dask = use_dask & default_da
        self.dask_chunks = dask_chunks

        self._learning_rate = learning_rate
        self._learning_rateN = learning_rateN

        if sigma == 0:
            self._sigma = min(x, y) / 2
        else:
            self._sigma = sigma

        self._std_coeff = std_coeff

        self._sigmaN = sigmaN
        self._input_len = input_len

        self.x = x
        self.y = y
        self.PBC = PBC
        self.n_nodes = x * y
        self.nodes = self.xp.arange(self.n_nodes)
        self.init = init

        if topology.lower() == "hexagonal":
            self.polygons = "Hexagons"
        else:
            self.polygons = "Squares"

        self.inner_dist_type = inner_dist_type

        self.neighborhood_function = neighborhood_function.lower()
        if self.neighborhood_function not in ["gaussian", "mexican_hat", "bubble"]:
            print(
                "{} neighborhood function not recognized.".format(self.neighborhood_function)
                + "Choose among 'gaussian', 'mexican_hat' or 'bubble'."
            )
            raise ValueError

        self.neighborhoods = Neighborhoods(
            self.x,
            self.y,
            self.polygons,
            self.inner_dist_type,
            self.PBC,
        )
        self.neighborhood_caller = partial(
            self.neighborhoods.neighborhood_caller,
            neigh_func=self.neighborhood_function,
        )

        decay_functions = {
            "exponential": exponential_decay,
            "asymptotic": asymptotic_decay,
            "linear": linear_decay,
        }

        if decay_function not in decay_functions:
            msg = "%s not supported. Functions available: %s"
            raise ValueError(msg % (decay_function, ", ".join(decay_functions.keys())))

        self._decay_function = decay_functions[decay_function]

        self.compact_support = compact_support

        self._activation_distance_name = activation_distance
        self._activation_distance_kwargs = activation_distance_kwargs
        self._activation_distance = DistanceFunction(
            activation_distance, activation_distance_kwargs, xp=self.xp
        )

        self._unravel_precomputed = self.xp.unravel_index(
            self.xp.arange(x * y, dtype=self.xp.int64), (x, y)
        )

        if n_parallel == 0:
            if self.xp.__name__ == "cupy":
                n_parallel = find_max_cuda_threads()
            else:
                n_parallel = find_cpu_cores() * DEFAULT_CPU_CORE_OVERSUBSCRIPTION

            if n_parallel == 0:
                raise ValueError(
                    "n_parallel was not specified and could not be infered from system"
                )

        self._n_parallel = n_parallel

        self._sq_weights_gpu = None

    def _init_weights(self, data) -> None:
        if self.init == "pca":
            init_vec = PCA(2).fit_transform(data)
        else:
            init_vec = [
                data.min(axis=0),
                data.max(axis=0),
            ]
        self.weights = (
            init_vec[0][None, :]
            + (init_vec[1] - init_vec[0])[None, :]
            * np.random.rand(self.n_nodes, *init_vec[0].shape)
        ).astype(np.float32)

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self.weights

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self.neighborhood.coordinates

    def activate(self, x):
        """Returns the activation map to x."""
        x_gpu = self.xp.array(x)
        weights_gpu = self.xp.array(self.weights)

        self._activate(x_gpu, weights_gpu)

        if GPU_SUPPORTED and isinstance(self._activation_map_gpu, cp.ndarray):
            return self._activation_map_gpu.get()
        else:
            return self._activation_map_gpu

    def _activate(self, x_gpu, weights_gpu):
        """Updates matrix activation_map, in this matrix
        the element i,j is the response of the neuron i,j to x"""
        if len(x_gpu.shape) == 1:
            x_gpu = self.xp.expand_dims(x_gpu, axis=0)

        if self._sq_weights_gpu is not None:
            self._activation_map_gpu = self._activation_distance(
                x_gpu, weights_gpu, self._sq_weights_gpu, xp=self.xp
            )
        else:
            self._activation_map_gpu = self._activation_distance(
                x_gpu, weights_gpu, xp=self.xp
            )

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError("num_iteration must be > 1")

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = "Received %d features, expected %d." % (data_len, self._input_len)
            raise ValueError(msg)
        
    def _update_rates(self, iteration, num_epochs):
        self.learning_rate = self._decay_function(
            self._learning_rate, self._learning_rateN, iteration, num_epochs
        )
        self.sigma = self._decay_function(self._sigma, self._sigmaN, iteration, num_epochs)

    def _winner(self, x_gpu, weights_gpu):
        """Computes the index of the winning neuron for the sample x"""
        if len(x_gpu.shape) == 1:
            x_gpu = self.xp.expand_dims(x_gpu, axis=0)

        self._activate(x_gpu, weights_gpu)
        raveled_idxs = self._activation_map_gpu.argmin(axis=1)
        return raveled_idxs

    def _update(self, x_gpu, weights_gpu):
        
        pre_numerator = self.xp.zeros(self.weights.shape, dtype=self.xp.float32)

        weights_gpu = self.xp.asarray(weights_gpu) # (X * Y, len)

        winners = self._winner(x_gpu, weights_gpu) # (N), xp
        
        series = winners[:, None] == self.nodes[None, :] # (N, X * Y), xp
        pop = self.xp.sum(series, axis=0, dtype=np.float32)
        
        for i, s in enumerate(series.T):
            pre_numerator[i, :] = self.xp.sum(x_gpu[s], axis=0)
        
        h = self.xp.asarray(self.neighborhood_caller(sigma=self.sigma)) # (X * Y, X * Y), xp

        _numerator_gpu = self.xp.dot(h, pre_numerator)
        _denominator_gpu = self.xp.dot(h, pop)[:, None]

        return (_numerator_gpu, _denominator_gpu)

    def _merge_updates(self, weights_gpu, numerator_gpu, denominator_gpu):
        """
        Divides the numerator accumulator by the denominator accumulator
        to compute the new weights.
        """
        return self.xp.where(
            denominator_gpu != 0, numerator_gpu / denominator_gpu, weights_gpu
        )

    def train(self, data, num_epochs, iter_beg=0, iter_end=None, verbose=False):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_epochs : int
            Maximum number of epochs (one epoch = all samples).
            In the code iteration and epoch have the same meaning.

        iter_beg : int, optional (default=0)
            Start from iteration at index iter_beg

        iter_end : int, optional (default=None, i.e. num_epochs)
            End before iteration iter_end (excluded) or after num_epochs
            if iter_end is None.

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        
        self._init_weights(data)
        
        if iter_end is None:
            iter_end = num_epochs

        # Copy arrays to device
        weights_gpu = self.xp.asarray(self.weights, dtype=self.xp.float32)

        if GPU_SUPPORTED and isinstance(data, cudf.core.dataframe.DataFrame):
            data_gpu = data.to_cupy(dtype=self.xp.float32)
            if self.use_dask:
                data_gpu_block = da.from_array(data_gpu, chunks=self.dask_chunks)
        elif GPU_SUPPORTED and isinstance(data, cp._core.core.ndarray):
            data_gpu = data.astype(self.xp.float32)
            if self.use_dask:
                data_gpu_block = da.from_array(data_gpu, chunks=self.dask_chunks)
        elif default_da and isinstance(data, ddf.core.DataFrame):
            if self.use_dask:
                data_gpu_block = data.to_dask_array()
            else:
                data_gpu = data.to_dask_array().compute()
        elif GPU_SUPPORTED and isinstance(data, dcudf.core.DataFrame):
            if self.use_dask:
                data_gpu = data.to_dask_array()
            data_gpu = data.compute()
        elif default_da and isinstance(data, da.core.Array):
            if self.use_dask:
                data_gpu_block = data
            else:
                data_gpu = data.compute().astype(self.xp.float32)
        else:
            data_gpu = self.xp.asarray(data, dtype=self.xp.float32)

        if verbose:
            print_progress(-1, num_epochs * len(data))

        for iteration in range(iter_beg, iter_end):
            try:  # reuse already allocated memory
                numerator_gpu.fill(0)
                denominator_gpu.fill(0)
            except UnboundLocalError:  # whoops, I haven't allocated it yet
                numerator_gpu = self.xp.zeros(weights_gpu.shape, dtype=self.xp.float32)
                denominator_gpu = self.xp.zeros(weights_gpu.shape, dtype=self.xp.float32)

            if self._activation_distance.can_cache:
                self._sq_weights_gpu = self.xp.power(weights_gpu, 2).sum(axis=1, keepdims=True)
            else:
                self._sq_weights_gpu = None

            self._update_rates(iteration, num_epochs)

            if self.use_dask:
                blocks = data_gpu_block.to_delayed().ravel()

                numerator_gpu_array = []
                denominator_gpu_array = []
                for block in blocks:
                    a, b = dask.delayed(self._update, nout=2)(
                        block, weights_gpu
                    )
                    numerator_gpu_array.append(a)
                    denominator_gpu_array.append(b)

                numerator_gpu_sum = dask.delayed(sum)(numerator_gpu_array)
                denominator_gpu_sum = dask.delayed(sum)(denominator_gpu_array)

                numerator_gpu, denominator_gpu = dask.compute(
                    numerator_gpu_sum, denominator_gpu_sum
                )
            else:
                for i in range(0, len(data), self._n_parallel):
                    start = i
                    end = start + self._n_parallel
                    if end > len(data):
                        end = len(data)

                    a, b = self._update(data_gpu[start:end], weights_gpu)

                    numerator_gpu += a
                    denominator_gpu += b

                    if verbose:
                        print_progress(
                            iteration * len(data) + end - 1, num_epochs * len(data)
                        )

            new_weights = self._merge_updates(
                weights_gpu, numerator_gpu, denominator_gpu
            )
            
            weights_gpu = (1 - self.learning_rate) * weights_gpu + self.learning_rate * new_weights
            weights_gpu = weights_gpu.astype(np.float32)

        # Copy back arrays to host
        if GPU_SUPPORTED and isinstance(weights_gpu, cp.ndarray):
            self.weights = weights_gpu.get()
        else:
            self.weights = weights_gpu

        # free temporary memory
        self._sq_weights_gpu = None

        if hasattr(self, "_activation_map_gpu"):
            del self._activation_map_gpu

        if verbose:
            print("\n quantization error:", self.quantization_error(data))

        return self

    def train_batch(self, data, num_iteration, verbose=False):
        """Compatibility with MiniSom, alias for train"""
        return self.train(data, num_iteration, verbose=verbose)

    def train_random(self, data, num_iteration, verbose=False):
        """Compatibility with MiniSom, alias for train"""
        print(
            "WARNING: due to batch SOM algorithm, random order is not supported. Falling back to train_batch."
        )
        return self.train(data, num_iteration, verbose=verbose)
    
    def predict(self, x):
        """Computes the indices of the winning neurons for the samples x."""

        if self.use_dask:
            x_gpu = da.from_array(self.xp.array(x))
        else:
            x_gpu = self.xp.array(x)

        weights_gpu = self.xp.array(self.weights)

        orig_shape = x_gpu.shape
        if len(orig_shape) == 1:
            if isinstance(x_gpu, da.core.Array):
                x_gpu = da.expand_dims(x_gpu, axis=0).compute()
            else:
                x_gpu = self.xp.expand_dims(x_gpu, axis=0)

        winners_chunks = []
        for i in range(0, len(x), self._n_parallel):
            start = i
            end = start + self._n_parallel
            if end > len(x):
                end = len(x)

            chunk = self._winner(x_gpu[start:end], weights_gpu)
            winners_chunks.append(chunk)

        winners_gpu = self.xp.hstack(winners_chunks)

        if GPU_SUPPORTED and isinstance(winners_gpu, cp.ndarray):
            winners = winners_gpu.get()
        else:
            winners = winners_gpu

        return winners

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""

        data_gpu = self.xp.array(data)
        qnt = self._quantization(data_gpu, self.xp.array(self.weights))

        if GPU_SUPPORTED and isinstance(qnt, cp.ndarray):
            return qnt.get()
        else:
            return qnt

    def _quantization(self, data_gpu, weights_gpu):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data_gpu)

        quantized = []
        for start in range(0, len(data_gpu), self._n_parallel):
            end = start + self._n_parallel
            winners_coords = self.xp.argmin(
                self._distance_from_weights(data_gpu[start:end], weights_gpu), axis=1
            )
            unraveled_indexes = self.xp.unravel_index(
                winners_coords, self.weights.shape[:2]
            )
            weights_gpu = self.xp.array(self.weights)
            quantized.append(weights_gpu[unraveled_indexes])

        return self.xp.vstack(quantized)

    def distance_from_weights(self, data, weights_gpu):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        data_gpu = self.xp.array(data)
        weights_gpu = self.xp.array(self.weights)
        d = self._distance_from_weights(data_gpu, weights_gpu)

        if GPU_SUPPORTED and isinstance(d, cp.ndarray):
            return d.get()
        else:
            return d

    def _distance_from_weights(self, data_gpu, weights):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        distances = []
        for start in range(0, len(data_gpu), self._n_parallel):
            end = start + self._n_parallel
            if end > len(data_gpu):
                end = len(data_gpu)
            w_flat = weights.reshape(-1, weights.shape[2])
            distances.append(
                euclidean_distance(data_gpu[start:end], w_flat, xp=self.xp)
            )
        return self.xp.vstack(distances)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)

        if self.use_dask:
            if default_da and isinstance(data, da.core.Array):
                data_gpu = data
            else:
                data_gpu = da.from_array(
                    self.xp.array(data, dtype=self.xp.float32), chunks=self.dask_chunks
                )

            blocks = data_gpu

            def _quantization_error_block(block, weights):
                weights_gpu = self.xp.array(weights)

                new_block = block - self._quantization(block, weights_gpu)

                return new_block

            q_error = blocks.map_blocks(
                _quantization_error_block, self.weights, dtype=self.xp.float32
            )

            qe_lin = da.linalg.norm(q_error, axis=1)
            qe = qe_lin.mean().compute()
        else:
            # load to GPU
            data_gpu = self.xp.array(data, dtype=self.xp.float32)
            weights_gpu = self.xp.array(self.weights)

            # recycle buffer
            data_gpu -= self._quantization(data_gpu, weights_gpu)

            qe = self.xp.linalg.norm(data_gpu, axis=1).mean()

        return qe.item()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not ajacent conunts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = np.prod(self.weights.shape)
        if total_neurons == 1:
            warn("The topographic error is not defined for a 1-by-1 map.")
            return np.nan

        # load to GPU
        data_gpu = self.xp.array(data, dtype=self.xp.float32)

        weights_gpu = self.xp.array(self.weights)

        distances = self._distance_from_weights(data_gpu, weights_gpu)

        # b2mu: best 2 matching units
        b2mu_inds = self.xp.argsort(distances, axis=1)[:, :2]
        b2my_xy = self.xp.unravel_index(b2mu_inds, self.weights.shape[:2])
        if self.topology == "rectangular":
            b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
            diff_b2mu_x = self.xp.abs(self.xp.diff(b2mu_x))
            diff_b2mu_y = self.xp.abs(self.xp.diff(b2mu_y))
            return ((diff_b2mu_x > 1) | (diff_b2mu_y > 1)).mean().item()
        elif self.topology == "hexagonal":
            b2mu_x = self._xx[b2my_xy[0], b2my_xy[1]]
            b2mu_y = self._yy[b2my_xy[0], b2my_xy[1]]
            dxdy = self.xp.hstack([self.xp.diff(b2mu_x), self.xp.diff(b2mu_y)])
            distance = self.xp.linalg.norm(dxdy, axis=1)
            return (distance > 1.5).mean().item()

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data.
        TODO: unoptimized
        """
        self._check_input_len(data)
        it = np.nditer(self.weights[:, :, 0], flags=["multi_index"])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self.weights[it.multi_index] = data[rand_i]
            it.iternext()

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.

        TODO: unoptimized
        """
        if self._input_len == 1:
            msg = "The data needs at least 2 features for pca initialization"
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = (
                "PCA initialization inappropriate:"
                + "One of the dimensions of the map is 1."
            )
            warn(msg)
        pc_length, pc = np.linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(-pc_length)
        for i, c1 in enumerate(np.linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self._neigy))):
                self.weights[i, j] = c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]]

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance.
        TODO: unoptimized
        """
        um = np.zeros(
            (self.weights.shape[0], self.weights.shape[1], 8)
        )  # 2 spots more for hexagonal topology

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

        if self.topology == "hexagonal":
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                w_2 = self.weights[x, y]
                e = y % 2 == 0  # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (
                        x + i >= 0
                        and x + i < self.weights.shape[0]
                        and y + j >= 0
                        and y + j < self.weights.shape[1]
                    ):
                        w_1 = self.weights[x + i, y + j]
                        um[x, y, k] = np.linalg.norm(w_2 - w_1)

        um = um.sum(axis=2)
        return um / um.max()

    def activation_response(self, data):
        """
        Returns a matrix where the element i,j is the number of times
        that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        winners = self.predict(data)
        for win in winners:
            a[win] += 1
        return a

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j.
        """
        self._check_input_len(data)
        winmap = defaultdict(list)
        winners = self.predict(data)
        for x, win in zip(data, winners):
            winmap[win].append(x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.

        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError("data and labels must have the same length.")
        winmap = defaultdict(list)
        winners = self.predict(data)
        for win, l in zip(winners, labels):
            winmap[win].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["xp"]
        del state["neighborhoods"]
        del state["neighborhood_caller"]
        del state["_activation_distance"]
        state["xp_name"] = self.xp.__name__
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        try:
            if self.xp_name == "cupy":
                self.xp = cp
            elif self.xp_name == "numpy":
                self.xp = np
        except:
            self.xp = default_xp

        self.neighborhoods = Neighborhoods(
            self.x,
            self.y,
            self.polygons,
            self.inner_dist_type,
            self.PBC,
        )
        self.neighborhood_caller = partial(
            self.neighborhoods.neighborhood_caller,
            neigh_func=self.neighborhood_function,
        )
        self._activation_distance = DistanceFunction(
            self._activation_distance_name, self._activation_distance_kwargs, self.xp
        )
