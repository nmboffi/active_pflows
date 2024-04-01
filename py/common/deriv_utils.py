"""
Nicholas M. Boffi
8/1/23

This file contains functions for computing derivatives of vector-valued functions.
"""

import jax
import jax.numpy as np
import functools
from typing import Callable


def single_partial(
    f: Callable[[np.ndarray], np.ndarray],  # [N, d] -> [N, d]
    inp: np.ndarray,  # [N, d]
    i: int,  # [0, ..., N-1]
    j: int,  # [0, ..., d-1]
) -> float:  # [N, d]
    """Compute a single partial derivative of a function on particles."""
    unit_vector = np.zeros_like(inp)
    unit_vector = unit_vector.at[i, j].set(1.0)

    return jax.jvp(f, (inp,), (unit_vector,))[1][i, j]


@functools.partial(jax.jit, static_argnums=0)
def vector_div(
    f: Callable[[np.ndarray], np.ndarray],
    inp: np.ndarray,  # [N, d]
) -> np.ndarray:
    """Compute the divergence of a vector-valued function on particles.
    Store all 'components' of the divergence in a single array."""
    return np.sum(
        jax.vmap(
            jax.vmap(
                single_partial,
                in_axes=(None, None, None, 0),
            ),
            in_axes=(None, None, 0, None),
        )(f, inp, np.arange(inp.shape[0]), np.arange(inp.shape[1])),
        axis=-1,
    )


@functools.partial(jax.jit, static_argnums=0)
def scalar_div(
    f: Callable[[np.ndarray], np.ndarray],
    inp: np.ndarray,  # [N, d]
) -> np.ndarray:
    """Compute the divergence of a vector-valued function on particles."""
    return np.sum(vector_div(f, inp))
