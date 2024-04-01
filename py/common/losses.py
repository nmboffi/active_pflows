"""
Nicholas M. Boffi

Loss utils for MIPS simulations.
"""

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from typing import Tuple, Callable
from jax import jit, vmap, value_and_grad
from functools import partial
import haiku as hk
import optax


@jit
def compute_grad_norm(grads: hk.Params) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return np.linalg.norm(flat_params) / np.sqrt(flat_params.size)


@partial(jit, static_argnums=(2, 3))
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
    loss_func: Callable[[hk.Params], float],
    loss_func_args: Tuple = tuple(),
) -> Tuple[hk.Params, optax.OptState, float, hk.Params]:
    """Update the neural network.

    Args:
        params: Parameters to optimize over.
        opt_state: State of the optimizer.
        opt: Optimizer itself.
        loss_func: Loss function for the parameters.
    """
    loss_value, grads = value_and_grad(loss_func)(params, *loss_func_args)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, grads


@partial(
    jax.pmap,
    in_axes=(0, 0, None, None, 0),
    static_broadcasted_argnums=(2, 3),
    axis_name="data",
)
def pupdate(
    params: hk.Params,
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
    loss_func: Callable[[hk.Params], float],
    loss_func_args: Tuple = tuple(),
) -> Tuple[hk.Params, optax.OptState, float, hk.Params]:
    """Update the neural network using data paralellism.

    Args:
        params: Parameters to optimize over.
        opt_state: State of the optimizer.
        opt: Optimizer itself.
        loss_func: Loss function for the parameters.
    """
    loss_value, grads = jax.value_and_grad(loss_func)(params, *loss_func_args)
    loss_value = jax.lax.pmean(loss_value, axis_name="data")
    grads = jax.lax.pmean(grads, axis_name="data")
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, grads
