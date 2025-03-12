import numpy as np
from scipy.stats import linregress, mode
from collections.abc import Sequence

try:
    import cupy as cp

    default_xp = cp
    _cupy_available = True
except ModuleNotFoundError:
    default_xp = np
    _cupy_available = False

from .utils import _get, compute

def compute_transmat(bmus, n_nodes, step: int = 1, yearbreaks: int | Sequence = 92, xp=default_xp):
    if isinstance(yearbreaks, int):
        iterator = range(yearbreaks, len(bmus) + 1, yearbreaks)
    else:
        iterator = np.cumsum(yearbreaks)
    trans_mat = xp.zeros((n_nodes, n_nodes))
    start_point = 0
    for end_point in iterator:
        real_end_point = min(end_point, len(bmus) - 1)
        theseind = xp.vstack(
            [
                bmus[start_point : real_end_point - step],
                xp.roll(bmus[start_point:real_end_point], -step)[:-step],
            ]
        ).T
        theseind, counts = xp.unique(theseind, return_counts=True, axis=0)
        trans_mat[theseind[:, 0], theseind[:, 1]] += counts
        start_point = real_end_point
    trans_mat /= xp.sum(trans_mat, axis=1)[:, None]
    return _get(compute(trans_mat))


def compute_residence_time( # old, use the one in persistent_spells.ipynb
    indices,
    n_nodes,
    distances,
    smooth_sigma: float = 0.0,
    yearbreaks: int = 92,
    q: float = 0.95,
    xp = default_xp,
):
    all_lengths = []
    all_lenghts_flat = []
    for j in range(n_nodes):
        all_lengths.append([])
        all_lenghts_flat.append([])
    start_point = 0
    for end_point in range(yearbreaks, len(indices) + 1, yearbreaks):
        for j in range(n_nodes):
            all_lengths[j].append([0])
        real_end_point = min(end_point, len(indices) - 1)
        these_indices = indices[start_point:real_end_point]
        jumps = xp.where(distances[these_indices[:-1], these_indices[1:]] != 0)[0]
        beginnings = xp.append([0], jumps + 1)
        lengths = xp.diff(xp.append(beginnings, [yearbreaks]))
        if smooth_sigma != 0:
            series_distances = (distances[these_indices[beginnings], :][:, these_indices[beginnings]] <= smooth_sigma).astype(int)
            series_distances[xp.tril_indices_from(series_distances, k=-1)] = 0
            how_many_more = xp.argmax(xp.diff(series_distances, axis=1) == -1, axis=1)[:-1] - xp.arange(len(beginnings) - 1)
            for i in range(len(lengths) - 1):
                lengths[i] = xp.sum(lengths[i:i + how_many_more[i] + 1])
        for beginning, length in zip(beginnings, lengths):
            node = mode(these_indices[beginning : beginning + length])
            all_lengths[node][-1].append(length)
            all_lenghts_flat[node].append(length)
        start_point = real_end_point
    trend_lengths = []
    max_lengths = []
    mean_lengths = []
    pvalues = []
    for i in range(n_nodes):
        mean_lengths.append(xp.mean(all_lenghts_flat[i]))
        max_each_year = xp.asarray([xp.quantile(all_lengths_, q=q) for all_lengths_ in all_lengths[i]])
        max_lengths.append(xp.amax(max_each_year))
        mask = max_each_year != 0
        trend, _, _, pvalue, _ = linregress(xp.arange(len(all_lengths[i]))[mask], max_each_year[mask])
        trend_lengths.append(trend)
        pvalues.append(pvalue)
    mean_lengths = xp.asarray(mean_lengths)
    max_lengths = xp.asarray(max_lengths)
    trend_lengths = xp.asarray(trend_lengths)
    pvalues = xp.asarray(pvalues)
    return _get(mean_lengths), _get(max_lengths), _get(trend_lengths), _get(pvalues), all_lengths


def get_index_columns(
    df,
    potentials: tuple = (
        "member",
        "time",
        "cluster",
        "jet ID",
        "spell",
        "relative_index",
    ),
):
    index_columns = [ic for ic in potentials if ic in df.columns]
    return index_columns


def compute_autocorrelation(
    indices, 
    n_nodes,
    lag_max: int = 50,
    xp=default_xp,
):
    """
        TODO: handle yearbreak and variable yearbreaks for two step clustering, i.e. yearbreaks as Sequence
    """
    series = indices[None, :] == xp.arange(n_nodes)[:, None]
    autocorrs = []
    for i in range(lag_max):
        autocorrs.append(
            xp.diag(
                xp.corrcoef(series[:, i:], xp.roll(series, i, axis=1)[:, i:])[
                    : n_nodes, n_nodes :
                ]
            )
        )
    return _get(xp.asarray(autocorrs))