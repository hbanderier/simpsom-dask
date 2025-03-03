from typing import Union, Callable, Tuple, Any
from collections.abc import Sequence

import numpy as np


class Neighborhoods:
    """Container class with functions to calculate neihgborhoods."""

    def __init__(
        self,
        width: int,
        height: int,
        polygons: str,
        dist_type: str | Sequence[str] = "grid",
        PBC: bool = False,
    ) -> None:
        """Instantiate the Neighborhoods class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """
        self.width = width
        self.height = height
        self.n_nodes = width * height
        self.nodes = np.arange(self.n_nodes)
        self.PBC = PBC
        self.polygons = polygons
        self.distances = self.compute_distances(dist_type)

    def compute_distances(
        self, dist_type: str | Sequence[str] = "grid"
    ) -> np.ndarray:
        if not isinstance(dist_type, str):
            self.sub_dist_type = dist_type[1]
            self.dist_type = dist_type[0]
        elif self.polygons.lower() == 'hexagons' and dist_type == 'grid':
            self.dist_type = dist_type
            self.sub_dist_type = "chebyshev"
        elif dist_type == 'grid':
            self.dist_type = dist_type
            self.sub_dist_type = 'l1'
        else:
            self.dist_type = dist_type
            self.sub_dist_type = 'l2'
        self.coordinates = np.asarray(
            [[x, y] for y in range(self.height - 1, -1, -1) for x in range(self.width)]
        ).astype(np.float32)
        if self.polygons.lower() == "hexagons":
            oddrows = self.coordinates[:, 1] % 2 == 1
            self.coordinates[:, 1] *= np.sqrt(3) / 2
            self.coordinates[oddrows, 0] += 0.5
        if dist_type == "cartesian":
            return self.cartesian_distances(
                self.coordinates, self.coordinates
            )
        return self.grid_distance(self.nodes, self.nodes)

    def cartesian_distances(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        self.dx = np.abs(b[:, None, 0] - a[None, :, 0])
        self.dy = np.abs(b[:, None, 1] - a[None, :, 1])
        if self.PBC:
            maxdx = self.width
            if self.polygons.lower() == "hexagons":
                maxdy = self.height * np.sqrt(3) / 2
            else:
                maxdy = self.height
            maskx = self.dx > maxdx / 2
            masky = self.dy > maxdy / 2
            self.dx[maskx] = maxdx - self.dx[maskx]
            self.dy[masky] = maxdy - self.dy[masky]
        if self.sub_dist_type.lower() in ["l1", "manhattan"]:
            return self.dx + self.dy
        elif self.sub_dist_type.lower() in ["linf", "chebyshev"]:
            return np.amax([self.dx, self.dy], axis=0)
        return np.sqrt(self.dx ** 2 + self.dy ** 2)

    def hexagonal_grid_distance(
        self, xi: np.ndarray, yi: np.ndarray, xj: np.ndarray, yj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        dy = yj[None, :] - yi[:, None]
        dx = xj[None, :] - xi[:, None]
        corr = yj[None, :] // 2 - yi[:, None] // 2
        if self.PBC:
            maskx = np.abs(dx) > (self.width / 2)
            masky = np.abs(dy) > (self.height / 2)
            dx[maskx] = -np.sign(dx[maskx]) * (self.width - np.abs(dx[maskx]))
            dy[masky] = -np.sign(dy[masky]) * (
                self.height - np.abs(dy[masky])
            )
            corr[masky] = -np.sign(corr[masky]) * (
                self.height // 2 - np.abs(corr[masky])
            )
        dx = dx - corr
        return dx, dy

    def rectangular_grid_distance(
        self, xi: np.ndarray, yi: np.ndarray, xj: np.ndarray, yj: np.ndarray
    ) -> np.ndarray:
        dy = np.abs(yj[None, :] - yi[:, None])
        dx = np.abs(xj[None, :] - xi[:, None])
        if self.PBC:
            maskx = np.abs(dx) > (self.width / 2)
            masky = np.abs(dy) > (self.height / 2)
            dx[maskx] = self.width - dx[maskx]
            dy[masky] = self.height - dx[maskx]
        return dx, dy

    def grid_distance(
        self, i: np.ndarray, j: np.ndarray
    ) -> np.ndarray:
        ndim = 0
        i, j = np.asarray(i), np.asarray(j)
        nx, ny = self.width, self.height
        for input_ in [i, j]:
            ndim += input_.ndim
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        yi, xi = divmod(i, nx)
        yi = ny - yi - 1
        yj, xj = divmod(j, nx)
        yj = ny - yj - 1
        if self.polygons.lower() == "hexagons":
            self.dx, self.dy = self.hexagonal_grid_distance(xi, yi, xj, yj)
            mask = np.sign(self.dx) == np.sign(self.dy)
            all_dists = np.where(
                mask,
                np.abs(self.dx + self.dy),
                np.amax([np.abs(self.dx), np.abs(self.dy)], axis=0),
            )
        else:
            self.dx, self.dy = self.rectangular_grid_distance(xi, yi, xj, yj)
            if self.sub_dist_type.lower() in ["linf", "chebyshev"]:
                all_dists = np.amax([self.dx, self.dy], axis=0)
            else:
                all_dists = self.dx + self.dy
        if ndim == 0:
            return all_dists[0, 0]
        elif ndim == 1:
            return all_dists.flatten()
        return all_dists

    def gaussian(self, c: np.ndarray, denominator: float) -> np.ndarray:
        """Gaussian neighborhood function.

        Args:
            c (np.ndarray): center points.
            denominator (float): the 2sigma**2 value.

        Returns
            (np.ndarray): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            thetax = np.exp(-np.power(self.dx[c], 2) / denominator) / np.sqrt(denominator * np.pi)
            thetay = np.exp(-np.power(self.dy[c], 2) / denominator) / np.sqrt(denominator * np.pi)
            return thetax * thetay
        return np.exp(-np.power(self.distances[c], 2) / denominator) / np.sqrt(denominator * np.pi)

    def mexican_hat(self, c: np.ndarray, denominator: float) -> np.ndarray:
        """Mexican hat neighborhood function.

        Args:
            c (np.ndarray): center points.
            denominator (float): the 2sigma**2 value.

        Returns
            (np.ndarray): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            thetax = np.power(self.dx[c], 2)
            thetay = np.power(self.dy[c], 2)
            theta = thetax + thetay
        else:
            theta = np.power(self.distances[c], 2)
        return (1 - 2 * theta / denominator) * np.exp(- theta / denominator)

    def bubble(self, c: np.ndarray, threshold: float) -> np.ndarray:
        """Bubble neighborhood function.

        Args:
            c (np.ndarray): center points.
            threshold (float): the bubble threshold.
        Returns
            (np.ndarray): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            return (self.dx[c] < threshold).astype(np.float32) * (self.dy[c] < threshold).astype(np.float32)
        return (self.distances[c] < threshold).astype(np.float32)

    def neighborhood_caller(
        self,
        centers: np.ndarray | None = None,
        sigma: float = 0.1,
        neigh_func: str = "gaussian",
    ) -> np.ndarray:

        d = 2 * sigma ** 2
        if centers is None:
            centers = self.nodes

        if neigh_func == "gaussian":
            return self.gaussian(centers, denominator=d)
        elif neigh_func == "mexican_hat":
            return self.mexican_hat(centers, denominator=d)
        elif neigh_func == "bubble":
            return self.bubble(centers, threshold=sigma)
        else:
            print(
                "{} neighborhood function not recognized.".format(neigh_func)
                + "Choose among 'gaussian', 'mexican_hat' or 'bubble'."
            )
            raise ValueError
