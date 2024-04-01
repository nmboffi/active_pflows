"""
Nicholas M. Boffi
7/24/23
Utilities for spatial gridding.
"""


import jax
import jax.numpy as np
from typing import Tuple
import functools


@jax.jit
def find_grid_pt(
    x: np.ndarray,     # [d]
    xgrid: np.ndarray, # [n_grid_pts]
    ygrid: np.ndarray  # [n_grid_pts]
) -> Tuple[int, int]:
    """Given a query point x and (linear) grids xgrid and ygrid
    in each direction, find the indices for the query."""
    xx, yy = x
    ii = np.argmax(xgrid > xx) - 1
    jj = np.argmax(ygrid > yy) - 1

    return ii, jj


def fill_grid(
    n_grid_pts: int,
    value: float,
    ii: int,
    jj: int,
) -> np.ndarray: # [n_grid_pts-1, n_grid_pts-1]
    """Fill a grid point with a given value. Also populate the grid with 1.0,
    for use in computing the density, if a normalized spatial grid is of interest."""
    grid = np.zeros((n_grid_pts-1, n_grid_pts-1))
    return grid.at[ii, jj].set(value), grid.at[ii, jj].set(1.0)


@jax.jit
def grid_quantity(
    particle_values,   # [N]
    xs: np.ndarray,    # [N, d]
    xgrid: np.ndarray, # [n_grid_pts]
    ygrid: np.ndarray, # [n_grid_pts]
) -> np.ndarray:       # [n_grid_pts-1, n_grid_pts-1]
    """Given an array of values on particles, map those values
    to a two-dimensional square grid. Return the two-dimensional square grid
    and a grid of multiplicities.
    """
    iis, jjs = jax.vmap(find_grid_pt, in_axes=(0, None, None))(xs, xgrid, ygrid)
    gridded_values, gridded_pts = jax.vmap(fill_grid, in_axes=(None, 0, 0, 0))(
        xgrid.size, particle_values, iis, jjs
    )
    return np.sum(gridded_values, axis=0), np.sum(gridded_pts, axis=0)


@functools.partial(jax.jit, static_argnums=4)
def average_grid_quantity(
    particle_values,   # [T, N]
    xs: np.ndarray,    # [T, N, d]
    xgrid: np.ndarray, # [n_grid_pts].
    ygrid: np.ndarray, # [n_grid_pts].
    sum_rslts: bool
) -> np.ndarray:       # [n_grid_pts-1, n_grid_pts-1]
    grids, multiplicities = jax.vmap(grid_quantity, in_axes=(0, 0, None, None))(
        particle_values, xs, xgrid, ygrid
    )
    pool = np.sum if sum_rslts else np.mean
    return pool(grids, axis=0), pool(multiplicities, axis=0)
