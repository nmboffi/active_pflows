"""
Nicholas M. Boffi

Drifts for MIPS simulations.
"""

import jax
from jax import vmap
import jax.numpy as np
from typing import Tuple


def wrapped_diff(x: np.ndarray, y: np.ndarray, width: float) -> float:
    """Compute wrapped single-coordinate differences on a periodic
    domain [-width, width]."""
    d = x - y
    return d - 2 * width * np.rint(d / (2 * width))


compute_wrapped_diffs = vmap(
    vmap(wrapped_diff, in_axes=(0, None, None), out_axes=0),
    in_axes=(None, 0, None),
    out_axes=1,
)


def softplus(x: np.ndarray, beta: float) -> np.ndarray:
    return jax.nn.softplus(beta * x) / beta


def single_particle_elastic_interaction(
    xi: np.ndarray, xs: np.ndarray, r: float, N: int, L: float, beta: float
) -> np.ndarray:
    """Elastic interaction from the Marchetti paper."""
    particle_diffs = vmap(lambda xj: wrapped_diff(xi, xj, L))(xs)  # [N, d]
    diff_norms = np.linalg.norm(particle_diffs, axis=1)  # [N]
    diff_norms = np.where(diff_norms == 0, -1, diff_norms)  # [N]
    directions = particle_diffs / diff_norms[:, None]  # [N, d]
    diff_norms = np.where(diff_norms == -1, 0, diff_norms)  # [N]

    if beta == 0 or beta == None:
        raise NotImplementedError
    else:
        Fijs = softplus(2 * r - diff_norms, beta) * (diff_norms > 0)  # [N]

    return np.sum(Fijs[:, None] * directions, axis=0)  # [d]


def elastic_interaction(
    xs: np.ndarray, radii: np.ndarray, N: int, L: float, beta: float
) -> np.ndarray:
    """Elastic interaction from the Marchetti paper."""
    particle_diffs = compute_wrapped_diffs(xs, xs, L)

    # auto-diffable norm at zero
    is_zero = np.eye(N, dtype=bool)
    d = particle_diffs.shape[-1]
    particle_diffs = np.where(is_zero[:, :, None], np.ones(d), particle_diffs)
    diff_norms = np.linalg.norm(particle_diffs, axis=-1)
    diff_norms = np.where(is_zero, 0.0, diff_norms)

    directions = particle_diffs / (np.eye(N) + diff_norms)[:, :, None]
    radii_sums = radii[:, None] + radii[None, :]

    if beta == None:
        Fijs = (radii_sums - diff_norms) * (diff_norms < radii_sums) * (diff_norms > 0)
    else:
        Fijs = softplus(radii_sums - diff_norms, beta) * (diff_norms > 0)

    return Fijs[:, :, None] * directions


def mips(
    xgs: np.ndarray,
    v0: float,
    A: float,
    k: float,
    gamma: float,
    radii: np.ndarray,
    width: float,
    beta: float,
    N: int,
) -> np.ndarray:
    """MIPS generalization of the active swimmer."""
    xs, gs = np.split(xgs, 2)
    forces = elastic_interaction(xs, radii, N, width, beta)
    xdots = v0 * gs + k * np.sum(forces, axis=1) - A * xs
    gdots = -gamma * gs

    return np.concatenate((xdots, gdots))


def torus_project(xs: np.ndarray, width: float):
    return ((xs + width) % (2 * width)) - width


def step_mips_OU_EM(
    xgs: np.ndarray,  # [2N, d]
    dt: float,
    radii: np.ndarray,
    A: float,
    k: float,
    v0: float,
    N: int,
    d: int,
    eps: float,
    gamma: float,
    width: float,
    beta: float,
    noise: np.ndarray,  # [2*N, d]
) -> np.ndarray:
    """Exact integration in the velocity variable with an Euler step
    in the x variable for the OU MIPS dynamics."""
    del d

    # split into two variables
    xs, gs = np.split(xgs, 2)
    noise_x, noise_g = np.split(noise, 2)

    # step x (EM)
    forces = elastic_interaction(xs, radii, N, width, beta)
    xdots = v0 * gs + k * np.sum(forces, axis=1) - A * xs
    xnexts = torus_project(xs + dt * xdots + np.sqrt(2 * eps * dt) * noise_x, width)

    # step g (exact)
    gnexts = gs * np.exp(-dt * gamma) + np.sqrt((1 - np.exp(-2 * gamma * dt))) * noise_g

    # concatenate the result and output
    return np.concatenate((xnexts, gnexts))
