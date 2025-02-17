from typing import Union, Collection, Tuple, Literal, Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Colormap, ListedColormap, Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import RegularPolygon, FancyArrowPatch
from itertools import product
from scipy.interpolate import LinearNDInterpolator
from .neighborhoods import Neighborhoods
import colormaps

TEXTWIDTH_IN = 0.0138889 * 503.61377

mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["figure.titlesize"] = 11
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["text.usetex"] = False
mpl.rcParams["animation.ffmpeg_path"] = r"~/mambaforge/envs/env11/bin/ffmpeg"


def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def infer_direction(to_plot: Any) -> int:
    max_ = np.nanmax(to_plot)
    min_ = np.nanmin(to_plot)
    try:
        max_ = max_.item()
        min_ = min_.item()
    except AttributeError:
        pass
    sym = np.sign(max_) == - np.sign(min_)
    sym = sym and np.abs(np.log10(np.abs(max_)) - np.log10(np.abs(min_))) <= 2
    if sym:
        return 0
    return 1 if np.abs(max_) > np.abs(min_) else -1  


def tile(
    polygons: str,
    coor: Tuple[float],
    color: Tuple[float],
    edgecolor: Tuple[float] = None,
    alpha: float = 0.1,
    linewidth: float = 1.0,
) -> RegularPolygon:
    """Set the tile shape for plotting.

    Args:
        polygons (str): type of polygons, case-insensitive
        coor (tuple[float, float]): positon of the tile in the plot figure.
        color (tuple[float,...]): color tuple.
        edgecolor (tuple[float,...]): border color tuple.

    Returns:
        (matplotlib patch object): the tile to add to the plot.
    """
    if polygons.lower() == "rectangle":
        return RegularPolygon(
            coor,
            numVertices=4,
            radius=0.95 / np.sqrt(2),
            orientation=np.radians(45),
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    elif polygons.lower() == "hexagons":
        return RegularPolygon(
            coor,
            numVertices=6,
            radius=0.95 / np.sqrt(3),
            orientation=np.radians(0),
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    else:
        raise NotImplementedError("Only hexagons and rectangles")


def draw_polygons(
    polygons: str,
    fig: Figure,
    centers: Collection[float],
    feature: Collection[float],
    ax: Axes = None,
    numbering: bool = False,
    cmap: ListedColormap | None | str = None,
    norm: Normalize | None = None,
    edgecolors: Tuple[float] | Collection[Tuple] = None,
    alphas: Collection[float] | float | int = None,
    linewidths: Collection[float] | float | int = 1.0,
    discretify: bool = True,
) -> Axes:
    """Draw a grid based on the selected tiling, nodes positions and color the tiles according to a given feature.

    Args:
        polygons_class (str): type of polygons, case-insensitive
        fig (matplotlib figure object): the figure on which the grid will be plotted.
        centers (list, float): array containing couples of coordinates for each cell
            to be plotted in the Hexagonal tiling space.
        feature (list, float): array contaning informations on the weigths of each cell,
            to be plotted as colors.
        cmap (ListedColormap): a custom color map.

    Returns:
        ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
    """
    if ax is None:
        ax = fig.add_subplot(111, aspect="equal")
    centers = np.asarray(centers)
    xpoints = centers[:, 0]
    ypoints = centers[:, 1]
    patches = []

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    elif cmap is None:
        cmap = plt.get_cmap("viridis")

    if np.isnan(feature).all():
        # edgecolors = "#555555"
        discretify = False

    if isinstance(edgecolors, str) or (edgecolors is None) or (len(edgecolors) == 3):
        edgecolors = [edgecolors] * len(feature)

    if alphas is None:
        alphas = 1.0

    if isinstance(alphas, int | float):
        alphas = [alphas] * len(feature)

    if linewidths is None:
        linewidths = 0.0

    if isinstance(linewidths, int | float):
        linewidths = [linewidths] * len(feature)
        
    symmetric = infer_direction(np.nan_to_num(feature)) == 0
    if discretify:
        levels = MaxNLocator(7 if symmetric else 5, symmetric=symmetric).tick_values(np.nanmin(feature), np.nanmax(feature))
        norm = BoundaryNorm(levels, cmap.N)
    if norm is not None:
        colors = cmap(norm(feature))
    else:
        colors = cmap(feature)
    for x, y, color, ec, alpha, linewidth in zip(
        xpoints, ypoints, colors, edgecolors, alphas, linewidths
    ):
        patches.append(
            tile(
                polygons,
                (x, y),
                color=color,
                edgecolor=ec,
                alpha=alpha,
                linewidth=linewidth,
            )
        )

    pc = PatchCollection(patches, match_original=True, cmap=cmap, norm=norm)
    pc.set_array(np.array(feature))
    pc.set_alpha(alphas)
    ax.add_collection(pc)

    dy = 1 / np.sqrt(3) if polygons == "hexagons" else 1 / np.sqrt(2)
    ax.set_xlim(xpoints.min() - 0.5, xpoints.max() + 0.5)
    ax.set_ylim(ypoints.min() - dy, ypoints.max() + dy)
    ax.axis("off")

    if numbering:
        from matplotlib.colors import rgb_to_hsv
        
        vs = rgb_to_hsv(colors[:, :3])[:, -1]
        if norm is not None:
            try:
                cutoff = np.sort(np.unique(vs))[-2]
            except IndexError:
                cutoff = 0
        else:
            cutoff = np.quantile(vs, 0.8)
        for i, c in enumerate(centers):
            x, y = c
            color = "white" if (vs[i] <= cutoff) and not symmetric else "black"
            ax.text(x, y, f'${i + 1}$', va='center', ha='center', color=color, fontsize=10)
    return ax


def plot_map(
    centers: Collection[np.ndarray],
    feature: Collection[np.ndarray],
    polygons: str,
    fig: Figure | None = None,
    ax: Axes | None = None,
    draw_cbar: bool = True,
    cbar_kwargs: dict | None = None,
    numbering: bool = False,
    **kwargs: Tuple
) -> Tuple[Figure, Axes]:
    """Plot a 2D SOM

    Args:
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        feature (list or array): The SOM node feature defining the color map
            (e.g. node.weights, node.diff)
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.
            - norm (Normalize): a Normalizer

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 4)
    if "cbar_label" in kwargs:
        cbar_kwargs["label"] = kwargs["cbar_label"]  # backwards compatibility baby
    if (cbar_kwargs or draw_cbar) and "shrink" not in cbar_kwargs:
        cbar_kwargs["shrink"] = 0.8

    if fig is None:
        fig, ax = plt.subplots(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]), subplot_kw=dict(aspect="equal"))
    ax = draw_polygons(
        polygons,
        fig,
        centers,
        feature,
        ax,
        numbering=numbering,
        cmap=kwargs.get("cmap", colormaps.matter),
        norm=kwargs.get("norm"),
        edgecolors=kwargs.get("edgecolors"),
        alphas=kwargs.get("alphas"),
        linewidths=kwargs.get("linewidths"),
        discretify=kwargs.get("discretify", 0),
    )

    if not np.isnan(feature).all() and (draw_cbar or cbar_kwargs):
        ticks = cbar_kwargs.pop("ticks", None)
        ticklabels = cbar_kwargs.pop("ticklabels", None)
        if ticks is not None:
            cbar_kwargs = cbar_kwargs | {"ticks": []}
        cbar = plt.colorbar(ax.collections[0], ax=ax, **cbar_kwargs)
        if ticks is not None:
            cbar.set_ticks(ticks, labels=ticklabels)        
        cbar.outline.set_visible(False)

    return fig, ax


def create_outer_grid(nx: int, ny: int, polygons: str = "hexagons") -> Tuple[np.ndarray]:
    nei = Neighborhoods(nx + 8, ny + 8, polygons, PBC=False)
    othernei = Neighborhoods(nx, ny, polygons, PBC=True)
    coords = nei.coordinates
    outer_grid = np.arange(nei.width * nei.height).reshape(
        nei.height, nei.width, order="C"
    )[::-1, :]
    outermask = np.zeros_like(outer_grid, dtype=bool)
    outermask[:4, :] = True
    outermask[-4:, :] = True
    outermask[:, :4] = True
    outermask[:, -4:] = True
    outermask = outermask[::-1, :].flatten(order="C")
    inner_grid = np.arange(othernei.width * othernei.height).reshape(
        othernei.height, othernei.width, order="C"
    )[::-1, :]
    slicex = slice(
        inner_grid.shape[0] - 4, inner_grid.shape[0] - 4 + outer_grid.shape[0]
    )
    slicey = slice(
        inner_grid.shape[1] - 4, inner_grid.shape[1] - 4 + outer_grid.shape[1]
    )
    outer_grid[:, :4] = np.tile(inner_grid[:, -4:].T, 5)[:, slicex].T
    outer_grid[:, -4:] = np.tile(inner_grid[:, :4].T, 5)[:, slicex].T
    outer_grid[:4, :] = np.tile(inner_grid[-4:, :], 5)[:, slicey]
    outer_grid[-4:, :] = np.tile(inner_grid[:4, :], 5)[:, slicey]
    outer_grid = outer_grid[::-1, :].flatten(order="C")
    outer_grid[~outermask] = inner_grid[::-1, :].flatten(order="C")
    return outer_grid, inner_grid, coords, outermask


def traj_to_segments(
    traj_split: np.ndarray,
    coords: np.ndarray,
    grid: np.ndarray,
    outermask: np.ndarray,
) -> Tuple[np.ndarray, list]:
    segments = []
    reps = []
    prev = coords[~outermask][traj_split[0][-1]]
    for k, j_ in enumerate(traj_split[1:]):
        j = j_[0]
        ks = np.where(grid == j)[0]
        distances = np.linalg.norm(prev[None, :] - coords[ks, :], axis=1)
        mindist = np.amin(distances)
        winnerks = ks[np.where(np.isclose(distances, mindist))[0]]
        next_ = coords[~outermask][j].copy()
        if np.all(outermask[winnerks]):
            winnerk = winnerks[0]
            segments.append(np.vstack([prev, coords[winnerk]]))
            ks = np.where(grid == traj_split[k][0])[0]
            distances = np.linalg.norm(next_[None, :] - coords[ks, :], axis=1)
            mindist = np.amin(distances)
            winnerk2 = ks[
                np.where(np.isclose(distances, mindist) & outermask[ks])[0][0]
            ]
            segments.append(np.vstack([coords[winnerk2], next_]))
            reps.append(2)
        else:
            segments.append(np.vstack([prev, next_]))
            reps.append(1)
        prev = next_.copy()

    segments = np.asarray(segments)
    return segments, reps


def segments_to_arcs(segments: np.ndarray, n_points: int = 50) -> Tuple[np.ndarray, list]:
    midpoints = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
    tangents = 0.5 * (segments[:, 1, :] - segments[:, 0, :])
    norm_tangents = np.linalg.norm(tangents, axis=-1)
    r = norm_tangents * 5
    tangents_perp = tangents[:, [1, 0]] * np.asarray([[1, -1]])
    centers = (
        midpoints
        - np.sqrt(r[:, None] ** 2 / norm_tangents[:, None] ** 2 - 1) * tangents_perp
    )
    correction = segments[:, :, 0] - centers[:, None, 0] < 0
    correction = np.pi * correction.astype(float)
    endpoints1 = (
        np.arctan(
            (segments[:, :, 1] - centers[:, None, 1])
            / (segments[:, :, 0] - centers[:, None, 0])
        )
        - correction
    )
    endpoints2 = (
        np.arctan(
            (segments[:, :, 1] - centers[:, None, 1])
            / (segments[:, :, 0] - centers[:, None, 0])
        )
        + correction
    )

    d1 = np.abs(endpoints1[:, 1] - endpoints1[:, 0])
    d2 = np.abs(endpoints2[:, 1] - endpoints2[:, 0])
    endpoints = np.where((d1 < d2)[:, None], endpoints1, endpoints2)
    ts = (
        endpoints[:, 0, None]
        + (endpoints[:, 1, None] - endpoints[:, 0, None])
        * np.linspace(0, 1, n_points)[None, :]
    )
    arcs = centers[:, None, :] + np.asarray(
        [r[:, None] * np.cos(ts), r[:, None] * np.sin(ts)]
    ).transpose(1, 2, 0)

    x1_arrow, y1_arrow = arcs[:, n_points // 2 - 3, :].T
    x2_arrow, y2_arrow = arcs[:, n_points // 2 + 3, :].T
    dx_arrow = x2_arrow - x1_arrow
    dy_arrow = y2_arrow - y1_arrow
    ds_arrow = np.sqrt(dx_arrow**2 + dy_arrow**2)
    dx_arrow = dx_arrow / ds_arrow * 0.15
    dy_arrow = dy_arrow / ds_arrow * 0.15
    arrows = []
    for x1, y1, dx, dy in zip(x1_arrow, y1_arrow, dx_arrow, dy_arrow):
        arrows.append(
            FancyArrowPatch(
                [x1, y1],
                [x1 + dx, y1 + dy],
                arrowstyle="wedge,tail_width=0.1,shrink_factor=0.4",
            )
        )
    return arcs, arrows


def plt_traj_hotspell(
    width: int, height: int, hotspell, bmus_da, da_T_region = None
):
    n_nodes = width * height
    traj_da = bmus_da.loc[hotspell]
    traj = traj_da.values
    outer_grid, inner_grid, coords, outermask = create_outer_grid(width, height)
    edgecolors = np.full(len(coords), "black", dtype=object)
    edgecolors[outermask] = "gray"
    alphas = np.ones(len(coords))
    linewidths = np.ones(len(coords)) * 2
    linewidths[outermask] = 0.5

    populations = np.zeros_like(outer_grid)
    populations[outermask] = 0
    thesepops = np.sum(traj[:, None] == np.arange(n_nodes)[None, :], axis=0)
    populations[~outermask] = thesepops
    
    traj_split = np.split(traj, np.where((np.diff(traj) != 0))[0] + 1)
    sizes = np.asarray([len(stay) for stay in traj_split])
    uniques = np.asarray([stay[0] for stay in traj_split])
    color_array = np.asarray([0, *np.cumsum(sizes)], dtype=int)
    cmap = colormaps.cet_l_bmw1_r
    colors = cmap(np.linspace(0, 1, len(traj) + 1))[color_array]
    sort_like = np.argsort(sizes)[::-1]

    segments, reps = traj_to_segments(traj_split, coords, outer_grid, outermask)
    arcs, arrows = segments_to_arcs(segments)
    if da_T_region is None:
        nx = 3
        width_ratios = [1, 0.05, 0.05]
    else:
        nx = 4
        width_ratios = [1, 0.05, 0.15, 0.05]
    gs = plt.GridSpec(
        1,
        nx,
        width_ratios=width_ratios,
        wspace=0.01,
        left=0.01,
        right=0.9,
        bottom=0.01,
        top=0.99,
    )
    fig = plt.figure(figsize=(width * (1 + 0.4), height))
    ax = fig.add_subplot(gs[0], aspect="equal")
    ax_cbar = fig.add_subplot(gs[-1])
    if da_T_region is not None:
        ax_temp = fig.add_subplot(gs[-2], sharey=ax_cbar)

    kwargs = dict(
        cmap=mpl.colormaps["gray_r"],
        norm=Normalize(np.amin(thesepops), np.amax(thesepops)),
    )
    xlims = [
        np.amin(coords[~outermask][:, 0]) - 0.8,
        np.amax(coords[~outermask][:, 0]) + 0.8,
    ]
    ylims = [
        np.amin(coords[~outermask][:, 1]) - 1,
        np.amax(coords[~outermask][:, 1]) + 1,
    ]
    fig, ax = plot_map(
        coords,
        populations,
        "hexagons",
        draw_cbar=False,
        figsize=(15 * width / height, 13.5),
        show=False,
        edgecolors="black",
        cmap="Greys",
        alphas=alphas,
        linewidths=linewidths,
        fig=fig,
        ax=ax,
    )

    lc = LineCollection(arcs, colors=np.repeat(colors[1:-1], reps, axis=0), zorder=3)
    lc.set_linewidth(7)
    lc = ax.add_collection(lc)
    arrows = PatchCollection(arrows, zorder=9, edgecolor=None, facecolor="white")
    ax.add_collection(arrows)
    im = ScalarMappable(Normalize(0, len(traj) - 1), cmap)
    cbar = fig.colorbar(
        im,
        cax=ax_cbar,
        label=f"Time during the hotspell in {traj_da[0].time.dt.year.item()}",
    )
    new_uniques = uniques[sort_like]
    sizes = sizes[sort_like]
    colors = colors[:-1][sort_like]
    ax.scatter(*coords[~outermask][new_uniques].T, s=100 * sizes, c=colors, zorder=10)

    every = 4 * (len(traj) // 40 + 1)
    list_of_days = np.asarray(np.arange(0, len(traj), every))
    pretty_list_of_days = traj_da[::every].time.dt.strftime("%b %d").values
    ax_cbar.set_yticks(list_of_days, labels=pretty_list_of_days)
    ax_cbar.invert_yaxis()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if da_T_region is None:
        return fig, ax
    
    temperature_profile = da_T_region.loc[hotspell]

    y = np.linspace(0, ax_cbar.get_ylim()[0], len(temperature_profile))
    ax_temp.spines[["top", "left"]].set_visible(False)
    ax_temp.set_ylabel("Regional temperature anomaly [K]")
    ax_temp.plot(temperature_profile, y, color="k", lw=2)
    ax_temp.fill_betweenx(
        y,
        temperature_profile,
        0,
        temperature_profile <= 0,
        color="blue",
        alpha=0.6,
        interpolate=True,
    )
    ax_temp.fill_betweenx(
        y,
        temperature_profile,
        0,
        temperature_profile >= 0,
        color="red",
        alpha=1.0,
        interpolate=True,
    )
    ax_temp.set_xticks([-3, 0, 3])
    ax_temp.tick_params(
        axis="y",
        labelleft=False,
        left=False,
        right=True,
        labelright=False,
        length=4,
        direction="in",
    )
    ax_temp.invert_xaxis()
    ax_temp.grid(axis="x")

    if any(np.isin(traj, [29, 35, 28, 34])):
        left = 0.05
    else:
        left = 0.5
    # cax = fig.add_axes([left, 0.009, 0.2, 0.05])
    # im = ScalarMappable(**kwargs)
    # fig.colorbar(im, cax=cax, orientation="horizontal")
    # cax.set_xticks(
    #     [0, np.amax(thesepops)],
    #     labels=["$0$", f"${int(np.amax(thesepops))}/{len(traj)}$"],
    # )
    for i, c in enumerate(coords):
        x, y = c
        j = outer_grid.flatten()[i]
        textcolor = "black"
        if outermask[i]:
            fontsize = 11
        else:
            if j in uniques and (uniques.tolist().index(j) / len(uniques) > 0.5):
                textcolor = "white"
            fontsize = 14
        if x > xlims[0] and x < xlims[-1] and y > ylims[0] and y < ylims[-1]: 
            ax.text(x, y, f'${1 + j}$', va='center', ha='center', color=textcolor, fontsize=fontsize, zorder=200)

    return fig, ax, cbar


def add_clusters(
    fig: Figure,
    ax: Axes,
    coords: np.ndarray,
    clu_labs: np.ndarray,
    polygons: str = "hexagons",
    cmap: str | Colormap | list | np.ndarray = None,
) -> Tuple[Figure, Axes]:
    unique_labs = np.unique(clu_labs)
    sym = np.any(unique_labs < 0)

    if cmap is None:
        cmap = "PiYG" if sym else "Greens"
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    nabove = np.sum(unique_labs > 0)
    if isinstance(cmap, list | np.ndarray):
        colors = cmap
    else:
        if sym:
            nbelow = np.sum(unique_labs < 0)
            cab = np.linspace(1, 0.66, nabove)
            cbe = np.linspace(0.33, 0, nbelow)
            if 0 in unique_labs:
                zerocol = [0.5]
            else:
                zerocol = []
            colors = [*cbe, *zerocol, *cab]
        else:
            if 0 in unique_labs:
                zerocol = [0.0]
            else:
                zerocol = []
            colors = np.linspace(1.0, 0.33, nabove)
            colors = [*zerocol, *colors]
        colors = cmap(colors)
    if polygons == "rectangle":
        dx, dy = coords[1, :] - coords[0, :]
        gen = [
            (sgnx * dx / 2.2, sgny * dy / 2.2)
            for sgnx, sgny in product([-1, 0, 1], [-1, 0, 1])
        ]
    elif polygons == "hexagons":
        l = 0.85 / np.sqrt(3)
        gen = [(l * degcos(theta), l * degsin(theta)) for theta in range(30, 360, 60)]
    for coord, val in zip(coords, clu_labs):
        newcs = [[coord[0] + cx, coord[1] + cy] for cx, cy in gen]
        coords = np.append(coords, newcs, axis=0)
        clu_labs = np.append(clu_labs, [val] * len(newcs))
    minx, miny = np.amin(coords, axis=0)
    maxx, maxy = np.amax(coords, axis=0)
    x = np.linspace(minx - 1, maxx + 1, 101)
    y = np.linspace(miny - 1, maxy + 1, 101)

    for i, lab in enumerate(np.unique(clu_labs)):
        interp = LinearNDInterpolator(coords, clu_labs == lab)
        r = interp(*np.meshgrid(x, y))
        if lab == 0:
            ax.contourf(x, y, r, levels=[0.8, 1], colors="black", alpha=0.6)
        else:
            ax.contour(x, y, r, levels=[0.8], colors=[colors[i]], linewidths=4)
    return fig, ax
