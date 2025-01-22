import datetime
from functools import partial
import numpy as np
import polars as pl
import xarray as xr
from scipy.stats import linregress, mode

try:
    import cupy as cp

    default_xp = cp
    _cupy_available = True
except:
    default_xp = np
    _cupy_available = False

from .utils import _get, compute

def compute_transmat(bmus, n_nodes, step: int = 1, yearbreaks: int = 92, xp=default_xp):
    trans_mat = xp.zeros((n_nodes, n_nodes))
    start_point = 0
    for end_point in range(yearbreaks, len(bmus) + 1, yearbreaks):
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


def xarray_to_polars(da: xr.DataArray | xr.Dataset):
    return pl.from_pandas(da.to_dataframe().reset_index())


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


def get_spells_sigma(df: pl.DataFrame, dists: np.ndarray, sigma: int = 1) -> pl.DataFrame:
    start = 0
    spells = []
    while True:
        start_lab = df[int(start), "labels"]
        next_distance_cond = dists[start_lab, df[start:, "labels"]] > sigma
        if not any(next_distance_cond):
            spells.append({"rel_start": start, "value": start_lab, "len": len(df[start:, "labels"])})
            break
        to_next = np.argmax(next_distance_cond)
        val = df[int(start): int(start + to_next), ["labels"]].with_columns(pl.col("labels").drop_nulls().mode().first().alias("mode"))[0, "mode"]
        spells.append({"rel_start": start, "value": val, "len": to_next})
        start = start + to_next
    return pl.DataFrame(spells).with_columns(year=df[0, "year"], my_len=df.shape[0])


def get_persistent_spell_times_from_som(
    labels, dists: np.ndarray, sigma: int = 0, minlen: int = 4, nt_before: int = 0, nt_after: int = 0, nojune: bool = True, daily: bool = False,
):
    labels_df = xarray_to_polars(labels)
    labels_df = labels_df.with_columns(pl.col("time").dt.year().alias("year"))
    index_columns = get_index_columns(labels_df)
    index = labels_df[index_columns].unique(maintain_order=True)

    out = labels_df.group_by("year", maintain_order=True).map_groups(partial(get_spells_sigma, dists=dists, sigma=sigma))
    out = out.with_columns(start = pl.col("year").rle_id() * pl.col("my_len") + pl.col("rel_start"))
    out = out.with_columns(
        range=pl.int_ranges(
            pl.col("start") - nt_before, pl.col("start") + pl.col("len") + nt_after
        ),
        relative_index=pl.int_ranges(-nt_before, pl.col("len") + nt_after, dtype=pl.Int16),
    )
    out = out.with_row_index("spell").explode(["range", "relative_index"])
    out = out.filter(pl.col("range") < len(index), pl.col("range") >= 0)
    out = out.with_columns(index[out["range"]])
    out = out.filter(pl.col("len") >= minlen)
    out = out.with_columns(pl.col("spell").rle_id())
    out = (
        out.group_by("spell", maintain_order=True)
        .agg(
            [
                pl.col(col).filter(
                    pl.col("time").dt.year()
                    == pl.col("time")
                    .dt.year()
                    .get(pl.arg_where(pl.col("relative_index") == 0).first())
                )
                for col in ["time", "relative_index", "value", "len"]
            ]
        )
        .explode(["time", "relative_index", "value", "len"])
    )
    if not nojune:
        return out
    june_filter = out.group_by("spell", maintain_order=True).agg(
        (pl.col("time").dt.ordinal_day() <= 160).sum() > 0.8
    )["time"]
    out = out.filter(pl.col("spell").is_in(june_filter.not_().arg_true()))
    out = out.with_columns(pl.col("spell").rle_id())
    out = out.with_columns(
        out.group_by("spell", maintain_order=True)
        .agg(
            relative_time=pl.col("time")
            - pl.col("time").get(pl.arg_where(pl.col("relative_index") == 0).first())
        )
        .explode("relative_time")
    )
    if not daily:
        return out
    
    ratio = out.filter(pl.col("relative_index") == 1)[0, "relative_time"] / datetime.timedelta(days=1)
    out = out.with_columns(pl.col("time").dt.round("1d")).unique(["spell", "time"], maintain_order=True)
    out = out.with_columns(out.group_by("spell", maintain_order=True).agg(pl.col("relative_index").rle_id() + (pl.col("relative_index").first() * ratio).round().cast(pl.Int16)).explode("relative_index"))
    out = out.with_columns(relative_time=pl.col("relative_index") * pl.duration(days=1))
    return out


def compute_autocorrelation(
    indices, 
    n_nodes,
    lag_max: int = 50,
    xp=default_xp,
):
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