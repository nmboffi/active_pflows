"""
Nicholas M. Boffi

Useful functions for computing EPRs.
"""

import jax.numpy as np
from jax import vmap
import haiku as hk
from typing import Tuple, Callable
from . import drifts


def d_softplus(x: float, beta: float):
    return 1.0 / (1.0 + np.exp(-x * beta))


def define_entropy(
    d: int,
    gamma: float,
    width: float,
    beta: float,
    r: float,
    apply_particle_div: Callable,
) -> Tuple[Callable, Callable, Callable]:
    def div_particle_drift(xs: np.ndarray, ii: int) -> float:  # [N, d]
        r"""Compute \nabla_i \cdot b_i."""
        xi = xs[ii]
        diffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
        rijs = np.linalg.norm(diffs, axis=1)

        ## discontinuous force
        if beta == None:
            rslts = 2 * r * (d - 1) / rijs - d
            rslts = rslts.at[ii].set(0.0)  # fix divide-by-zero
            return np.sum(rslts * (rijs < 2 * r))

        ## smoothed, softplus force
        else:
            rslts = drifts.softplus(2 * r - rijs, beta) * (d - 1) / rijs - d_softplus(
                2 * r - rijs, beta
            )
            rslts = rslts.at[ii].set(0.0)  # fix divide-by-zero
            return np.sum(rslts)

    def single_particle_entropy(
        params: hk.Params, xgs: np.ndarray, ii: int, noise_free: bool  # [2N, d]
    ) -> float:
        xs, gs = np.split(xgs, 2)

        if noise_free:
            return div_particle_drift(xs, ii)
        else:
            return -gamma * apply_particle_div(params, xgs, ii) - gamma * d

    in_axes = (None, None, 0, None)

    calc_particle_entropies = lambda params, xgs, batch_inds, noise_free: vmap(
        single_particle_entropy, in_axes
    )(params, xgs, batch_inds, noise_free)

    return div_particle_drift, single_particle_entropy, calc_particle_entropies
