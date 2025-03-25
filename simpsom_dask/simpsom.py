from functools import partial, wraps
from pathlib import Path
from warnings import warn
from collections import defaultdict, Counter
from typing import Sequence, Literal
from sys import stdout
from time import time
from datetime import timedelta

import numpy as np
from tqdm import trange
from sklearn.metrics import pairwise_distances # better optimized when x=y

try:
    # Cupy needs to be imported first.
    # Cudf is crashing containers if it goes first.
    import cupy as cp
    import cudf
    import dask_cudf as dcudf

    default_xp = cp
    GPU_SUPPORTED = True
except ModuleNotFoundError:
    default_xp = np
    GPU_SUPPORTED = False

try:
    import dask
    import dask.array as da
    import dask.delayed as dd
    import dask.dataframe as ddf
    from dask_ml.decomposition import PCA as da_PCA

    default_da = True
except ModuleNotFoundError:
    print("WARNING: Dask Arrays could not be imported")
    default_da = False

from sklearn.decomposition import PCA


from .distances import DistanceFunction, euclidean_distance
from .neighborhoods import Neighborhoods
from .utils import find_cpu_cores, find_max_cuda_threads, _get, compute, triangulize
from .decays import linear_decay, asymptotic_decay, exponential_decay
from .diagnostics import compute_residence_time, compute_autocorrelation, compute_transmat
from .plots import plot_map

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


class Simpsom:
    def __init__(
        self,
        x,
        y,
        sigma=0,
        sigmaN=1,
        decay_function="exponential",
        neighborhood_function="gaussian",
        std_coeff=0.5,
        topology="hexagonal",
        inner_dist_type: str | Sequence[str] = "grid",
        PBC: bool = True,
        activation_distance="euclidean",
        activation_distance_kwargs={},
        init: Literal["random", "pca"] = "random",
        random_seed=None,
        n_parallel=0,
        compact_support=False,
        xp=default_xp,
        use_dask=True,
        dask_chunks=(100, -1),
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

        decay_function : string, optional (default='exponential')
            Function that reduces sigma at each iteration.
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

        if sigma == 0:
            self._sigma = min(x, y) / 2
        else:
            self._sigma = sigma

        self._std_coeff = std_coeff

        self._sigmaN = sigmaN

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
                "{} neighborhood function not recognized.".format(
                    self.neighborhood_function
                )
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
        
    def load_weights(self, path: Path):
        weights = np.load(path)
        assert weights.shape[0] == self.n_nodes
        self.weights = weights
        self._input_len = self.weights.shape[1]

    def _coerce(self, array):
        """Turns numpy to cupy
        Turns non dask to dask
        If dask with numpy backend and should be dask with cupy backend, leaves as is.

        Args:
            array (_type_): _description_
        """
        if self.use_dask and isinstance(array, da.Array):
            return array
        if self.use_dask:
            return da.from_array(self.xp.array(array), chunks=self.dask_chunks)
        return self.xp.array(array)

    def _init_weights(self, data) -> None:
        rng = np.random.default_rng()
        if self.init == "pca":
            if isinstance(data, da.core.Array):
                pca = da_PCA
            else:
                pca = PCA
            print("PCA weights init")
            pca_res = pca(2).fit(data)
            init_vec = pca_res.components_
            trans = pca_res.transform(data[rng.integers(data.shape[0], size=10000)])
            mins = trans.max()
            maxs = trans.min()
            span_x = np.linspace(-mins, maxs, self.x) / 3
            span_y = np.linspace(-mins, maxs, self.y) / 3
            if self.PBC:
                span_x = triangulize(span_x)
                span_y = triangulize(span_y)
            span_x = span_x + rng.random(span_x.shape) * 0.1 * (maxs - mins)
            span_y = span_y + rng.random(span_y.shape) * 0.1 * (maxs - mins)
            wx = span_x[:, None] * init_vec[0][None, :]
            wy = span_y[:, None] * init_vec[1][None, :]
            self.weights = wx[:, None, :] + wy[None, :, :]
            self.weights = self.weights.reshape(-1, wx.shape[1])
            
        else:
            print("Random weights init")
            init_vec = [
                data[rng.integers(len(data), size=1000)].min(axis=0),
                data[rng.integers(len(data), size=1000)].max(axis=0),
            ]
            self.weights = (
                init_vec[0][None, :]
                + (init_vec[1] - init_vec[0])[None, :]
                * rng.random((self.n_nodes, *init_vec[0].shape))
            ).astype(np.float32)
        self.weights = compute(self.weights, progress_flag=True)
        self._input_len = self.weights.shape[1]

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

        return _get(self._activation_map_gpu)

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
        self.sigma = self._decay_function(
            self._sigma, self._sigmaN, iteration, num_epochs
        )

    def _winner(self, x_gpu, weights_gpu):  #
        """Computes the index of the winning neuron for the sample x"""
        if len(x_gpu.shape) == 1:
            x_gpu = self.xp.expand_dims(x_gpu, axis=0)

        self._activate(x_gpu, weights_gpu)
        raveled_idxs = self._activation_map_gpu.argmin(axis=1)
        return raveled_idxs

    def _update(self, x_gpu, weights_gpu):

        pre_numerator = self.xp.zeros(self.weights.shape, dtype=self.xp.float32)

        weights_gpu = self.xp.asarray(weights_gpu)  # (X * Y, len)

        winners = self._winner(x_gpu, weights_gpu)  # (N), xp

        series = winners[:, None] == self.nodes[None, :]  # (N, X * Y), xp
        pop = self.xp.sum(series, axis=0, dtype=np.float32)

        for i, s in enumerate(series.T):
            pre_numerator[i, :] = self.xp.sum(x_gpu[s], axis=0)

        h = self.xp.asarray(
            self.neighborhood_caller(sigma=self.sigma)
        )  # (X * Y, X * Y), xp

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

    def train(self, data, num_epochs, iter_beg=0, iter_end=None, verbose=False, out_path: Path = None):
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
        elif default_da and isinstance(data, np.ndarray):
            if self.use_dask:
                data_gpu_block = da.from_array(data, chunks=self.dask_chunks)
            else:
                data_gpu = data
        elif default_da and isinstance(data, ddf.core.DataFrame):
            if self.use_dask:
                data_gpu_block = data.to_dask_array()
            else:
                data_gpu = compute(data.to_dask_array())
        elif GPU_SUPPORTED and isinstance(data, dcudf.core.DataFrame):
            if self.use_dask:
                data_gpu = data.to_dask_array()
            data_gpu = compute(data)
        elif default_da and isinstance(data, da.core.Array):
            if self.use_dask:
                data_gpu_block = data
            else:
                data_gpu = compute(data).astype(self.xp.float32)
        else:
            data_gpu = self.xp.asarray(data, dtype=self.xp.float32)

        if verbose:
            print_progress(-1, num_epochs * len(data))
        
        for iteration in trange(iter_beg, iter_end):
            try:  # reuse already allocated memory
                numerator_gpu.fill(0)
                denominator_gpu.fill(0)
            except UnboundLocalError:  # whoops, I haven't allocated it yet
                numerator_gpu = self.xp.zeros(weights_gpu.shape, dtype=self.xp.float32)
                denominator_gpu = self.xp.zeros(
                    weights_gpu.shape, dtype=self.xp.float32
                )

            if self._activation_distance.can_cache:
                self._sq_weights_gpu = self.xp.power(weights_gpu, 2).sum(
                    axis=1, keepdims=True
                )
            else:
                self._sq_weights_gpu = None

            self._update_rates(iteration, num_epochs)

            if self.use_dask:
                blocks = data_gpu_block.to_delayed().ravel()

                numerator_gpu_array = []
                denominator_gpu_array = []
                for block in blocks:
                    a, b = dask.delayed(self._update, nout=2)(block, weights_gpu)
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

            weights_gpu = self._merge_updates(
                weights_gpu, numerator_gpu, denominator_gpu
            )

        # Copy back arrays to host
        self.weights = _get(compute(weights_gpu))
        if out_path is not None:
            np.save(out_path, self.weights)
         
        self.latest_bmus = self.predict(data)

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
    
    def memoize(func):
        """Store the results of the decorated function for fast lookup
        """

        # Store results in a dict that maps arguments to results
        cache = {}
        @wraps(func)
        def wrap(self, x):
            # If these arguments haven't been seen before, call func() and store the result.
            shape = x.shape
            sum_ = compute(x[:, shape[1] // 2].sum())
            sum_ += self.weights[:, shape[1] // 2].sum()
            key = (*shape, sum_)
            if key not in cache:        
                cache[key] = cc = func(self, x)          
                return cc
            print("Memoized")
            return cache[key]
        return wrap
    
    def predict(self, x = None):
        """Computes the indices of the winning neurons for the samples x."""
        if x is None:
            try:
                return self.latest_bmus
            except AttributeError:
                print("Provide x at least once")
                raise ValueError
        self.latest_bmus = self._predict(x)
        return self.latest_bmus
    
    @memoize
    def _predict(self, x):
        x_gpu = self._coerce(x)
        weights_gpu = self.xp.asarray(self.weights)
        if not GPU_SUPPORTED:
            return compute(self._winner(x_gpu, weights_gpu), progress_flag=True) 

        orig_shape = x_gpu.shape
        if len(orig_shape) == 1:
            if isinstance(x_gpu, da.core.Array):
                x_gpu = compute(da.expand_dims(x_gpu, axis=0))
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

        return _get(compute(winners_gpu), progress_flag=True)

    def quantization(self, x = None):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""

        winners = compute(self.predict(x))
        weights = _get(self.weights)
        return weights[winners]

    def distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        data_gpu = self._coerce(data)
        weights_gpu = self.xp.array(self.weights)
        d = self._distance_from_weights(data_gpu, weights_gpu)
        return _get(d)

    def _distance_from_weights(self, data_gpu, weights_gpu):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        distances = []
        for start in range(0, len(data_gpu), self._n_parallel):
            end = start + self._n_parallel
            if end > len(data_gpu):
                end = len(data_gpu)
            distances.append(
                euclidean_distance(data_gpu[start:end], weights_gpu, xp=self.xp)
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

                new_block = block - weights_gpu[self._winner(block, weights_gpu)]

                return new_block

            q_error = blocks.map_blocks(
                _quantization_error_block, self.weights, dtype=self.xp.float32
            )

            qe_lin = da.linalg.norm(q_error, axis=1)
            qe = qe_lin.mean()
        else:
            # load to GPU
            data_gpu = self.xp.array(data, dtype=self.xp.float32)
            weights_gpu = self.xp.array(self.weights)

            # recycle buffer
            data_gpu -= self._quantization(data_gpu, weights_gpu)

            qe = self.xp.linalg.norm(data_gpu, axis=1).mean().item()

        return compute(qe)

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
        if self.n_nodes == 1:
            warn("The topographic error is not defined for a 1-by-1 map.")
            return np.nan

        # load to GPU
        data_gpu = self._coerce(data)

        weights_gpu = self.xp.array(self.weights)

        distances = self._distance_from_weights(data_gpu, weights_gpu)

        # b2mu: best 2 matching units
        if isinstance(distances, da.core.Array):
            b2mu_inds = compute(da.argtopk(distances, k=-2, axis=1))
        else:
            b2mu_inds = self.xp.argsort(distances, axis=1)[:, :2]

        grid_dists = self.neighborhoods.distances
        grid_dists = grid_dists[b2mu_inds[:, 0], b2mu_inds[:, 1]]
        topo_error = (grid_dists > 1.5).mean()
        try:
            return topo_error.item()
        except AttributeError:
            return topo_error

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance.
        TODO: unoptimized
        """
        weights_gpu = self._coerce(self.weights)
        weights_dist = euclidean_distance(weights_gpu, weights_gpu, xp=self.xp)
        pos_dist = self.neighborhoods.distances
        weights_dist[(pos_dist > 1.01) | (pos_dist == 0.0)] = np.nan
        return _get(compute(self.xp.nanmean(weights_dist)))

    def compute_populations(self, data = None):
        """
        Returns a matrix where the element i,j is the number of times
        that the neuron i,j have been winner.
        """
        winners = self.predict(data)
        return np.asarray([np.sum(winners == i) for i in range(self.n_nodes)])

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

    def compute_transmat(self, data = None, step: int = 1, yearbreaks: int | Sequence = 92):
        winners = compute(self.predict(data))
        return compute_transmat(
            winners, self.n_nodes, step=step, yearbreaks=yearbreaks, xp=self.xp
        )

    def compute_residence_time(
        self, data = None, smooth_sigma: float = 0.0, yearbreaks: int = 92, q: float = 0.95
    ):
        winners = compute(self.predict(data))
        return compute_residence_time(
            winners,
            self.n_nodes,
            self.neighborhoods.distances,
            smooth_sigma=smooth_sigma,
            yearbreaks=yearbreaks,
            q=q,
            xp=self.xp,
        )
    
    def compute_residence_time_real_sigma(
        self, data = None, smooth_sigma_quantile: float = 0.1, yearbreaks: int = 92, q: float = 0.95
    ):
        winners = compute(self.predict(data))
        pairwise = pairwise_distances(self.weights)
        smooth_sigma = np.quantile(pairwise[pairwise > 0], smooth_sigma_quantile)
        return compute_residence_time(
            winners,
            self.n_nodes,
            pairwise,
            smooth_sigma=smooth_sigma,
            yearbreaks=yearbreaks,
            q=q,
            xp=self.xp,
        )
        
    def compute_autocorrelation(self, data = None, lag_max: int = 50):
        winners = compute(self.predict(data))
        return compute_autocorrelation(
            winners, 
            self.n_nodes,
            lag_max=lag_max,
            xp=self.xp,
        )
        
    def smooth(
        self,
        data,
        smooth_sigma: float = 0,
        neigh_func: str | None = None,
    ):
        if np.isclose(smooth_sigma, 0.0):
            return data
        if neigh_func is None:
            neigh_func = self.neighborhood_function
        theta = self.neighborhoods.neighborhood_caller(
            np.arange(self.n_nodes), smooth_sigma, neigh_func=neigh_func
        )
        return _get(self.xp.sum((data[None, :] * theta), axis=1) / self.xp.sum(theta, axis=1))
    
    def plot_on_map(
        self,
        data,
        smooth_sigma: float = 0,
        fig = None,
        ax = None,
        draw_cbar: bool = True,
        **kwargs
    ):
        """Wrapper function to plot a trained 2D SOM map
        color-coded according to a given feature.

        Args:
            data (NDArray): What to show on the map.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        data = self.smooth(data, smooth_sigma)
        fig, ax = plot_map(
            self.neighborhoods.coordinates,
            data,
            self.polygons,
            fig=fig,
            ax=ax,
            draw_cbar=draw_cbar,
            **kwargs
        )

        return fig, ax

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
