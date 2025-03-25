<h1>XPySom-dask</h1>

Self Organizing Maps with Dask Support and pretty hexagonal plot.
--------------------

Simpsom-Dask is a partial rewrite of [XPySom-Dask](https://github.com/jcfaracco/xpysom-dask) with vizualisation elements introduced in [Simpsom](https://github.com/fcomitani/simpsom). Some optimizations have been added, and the vizualisation elements have been improved, mostly to fit my needs.

Installation
---------------------

You can download XPySom-dask from PyPi:

    pip install simpsom-dask

By default, dependencies for GPU execution are not downloaded. 

You can also choose to clone this repo and then run
    
    cd simpsom-dask
    pip install -e .

if you would like to modify things on the fly.

How to use it
---------------------

The module interface is similar to [MiniSom](https://github.com/JustGlowing/minisom.git). In the following only the basics of the usage are reported, for an overview of all the features, please refer to the original MiniSom examples you can refer to: https://github.com/JustGlowing/minisom/tree/master/examples (you can find the same examples also in this repository but they have not been updated yet).

In order to use Simpsom you need your data organized as a Numpy matrix where each row corresponds to an observation or as list of lists like the following:

```python
chunks = (4, 2)
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      
```

 Then you can train XPySom just as follows:

```python
from simsom_dask import Simpsom

import dask.array as da

from dask.distributed import Client, LocalCluster

client = Client(LocalCluster())

dask_data = da.from_array(data, chunks=chunks)

som = Simpsom(6, 6, 4, sigma=0.3, learning_rate=0.5, use_dask=True, chunks=chunks) # initialization of 6x6 SOM
som.train(dask_data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

Differences with MiniSom
---------------------
 - The batch SOM algorithm is used (instead of the online used in MiniSom). Therefore, use only `train` to train the SOM, `train_random` and `train_batch` are not present.
 - `decay_function` input parameter is no longer a function but one of `'linear'`,
 `'exponential'`, `'asymptotic'`. As a consequence of this change, `sigmaN` and `learning_rateN` have been added as input parameters to represent the values at the last iteration.
 - New input parameter `std_coeff`, used to calculate gaussian exponent denominator `d = 2*std_coeff**2*sigma**2`. Default value is 0.5 (as in [Somoclu](https://github.com/peterwittek/somoclu), which is **different from MiniSom original value** sqrt(pi)).
 - New input parameter `xp` (default = `cupy` module). Back-end to use for computations.
 - New input parameter `n_parallel` to set size of the mini-batch (how many input samples to elaborate at a time).
 - **Hexagonal** grid support is recommended.  


Authors
---------------------

Copyright (C) 2025 Hugo Banderier
