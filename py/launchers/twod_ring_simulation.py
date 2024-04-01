"""
NicH V Jagadishholas M. Boffi
7/19/23

Code for systematic simulations of two particles in one dimension on a ring.
"""

import jax
import jax.numpy as np
import numpy as onp
import dill as pickle
from typing import Tuple, Callable, Dict
from ml_collections import config_dict
from copy import deepcopy
import argparse


import time
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.auto import tqdm as tqdm
import wandb


import haiku as hk
from flax.jax_utils import replicate, unreplicate
import sys
import optax

sys.path.append("../../py")
import common.networks as networks
import common.losses as losses
import common.drifts as drifts
from typing import Callable, Tuple


######## sensible matplotlib defaults ########
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["grid.color"] = "0.8"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["figure.titlesize"] = 7.5
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.fontsize"] = 7.5
mpl.rcParams["figure.dpi"] = 300
#############################################


########## Dataset ###############
def calc_xdot(x: np.ndarray, g: np.ndarray) -> np.ndarray:  # [1, d]  # [1, d]  # [1, d]
    if cfg.shift_system:
        norm = np.min(np.array([np.linalg.norm(x - 2 * cfg.width), np.linalg.norm(x)]))
        xhat = -np.sign(x - cfg.width)
    else:
        norm = np.linalg.norm(x)
        xhat = x / norm

    if cfg.smooth_force:
        force = cfg.k * drifts.softplus(2 - norm, cfg.beta) * xhat
    else:
        force = cfg.k * (2 - norm) * xhat * (norm < 2)

    if cfg.clip_force:
        #        force = force * (np.linalg.norm(force) > cfg.min_force)
        force = force * (norm < cfg.max_distance)

    return cfg.v0 * g + force - cfg.A * x


def step_system(
    xg: np.ndarray,  # [2, d]
    noise: np.ndarray,  # [2, d]
) -> Tuple[np.ndarray, np.ndarray]:
    # split into two variables
    x, g = np.split(xg, 2)
    noise_x, noise_g = np.split(noise, 2)

    # step x
    xdot = calc_xdot(x, g)

    if cfg.shift_system:
        xnext = (x + cfg.dt * xdot + np.sqrt(2 * cfg.eps * cfg.dt) * noise_x) % (
            2 * cfg.width
        )
    else:
        xnext = drifts.torus_project(
            x + cfg.dt * xdot + np.sqrt(2 * cfg.eps * cfg.dt) * noise_x, cfg.width
        )

    # step g (exact)
    gnext = (
        g * np.exp(-cfg.dt * cfg.gamma)
        + np.sqrt((1 - np.exp(-2 * cfg.gamma * cfg.dt))) * noise_g
    )

    # concatenate the result and output
    return np.concatenate((xnext, gnext))


def rollout(
    init_xg: np.ndarray,  # [2, d]
    noises: np.ndarray,  # [nsteps, 2, d]
) -> np.ndarray:
    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = step_system(xg, noise)
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final, xg_traj


@jax.jit
def rollout_trajs(
    init_xgs: np.ndarray,  # [ntrajs, 2, d]
    noises: np.ndarray,  # [ntrajs, nsteps, 2, d]
) -> np.ndarray:
    return jax.vmap(lambda init_xg, traj_noises: rollout(init_xg, traj_noises)[0])(
        init_xgs, noises
    )


def generate_data(
    key: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset from the example problem."""
    if cfg.shift_system:
        xs = (cfg.width + cfg.sig0x * onp.random.randn(cfg.ntrajs, cfg.N, cfg.d)) % (
            2 * cfg.width
        )
    else:
        xs = drifts.torus_project(
            cfg.sig0x * onp.random.randn(cfg.ntrajs, cfg.N, cfg.d), cfg.width
        )

    gs = cfg.sig0g * onp.random.randn(cfg.ntrajs, cfg.N, cfg.d)
    xgs = onp.concatenate((xs, gs), axis=1)
    nbatches_rollout = int(cfg.nsteps / cfg.max_n_steps) + 1

    start_time = time.time()
    print(f"Starting data generation.")
    for curr_batch in range(nbatches_rollout):
        noises = jax.random.normal(
            key, shape=(cfg.ntrajs, cfg.max_n_steps, 2 * cfg.N, cfg.d)
        )
        xgs = rollout_trajs(xgs, noises)
        key = jax.random.split(key)[0]
    end_time = time.time()
    print(f"Finished data generation. Total time={(end_time-start_time)/60.}m")

    return symmetrize_data(onp.array(xgs)), key


def symmetrize_data(xgs: np.ndarray) -> np.ndarray:
    """Symmetrize the data."""
    if cfg.shift_system:
        xs, gs = onp.split(xgs, 2, axis=1)
        xgs_symm = onp.concatenate((2 * cfg.width - xs, -gs), axis=1)
        symmetrized = onp.concatenate((xgs, xgs_symm), axis=0)
    else:
        symmetrized = onp.concatenate((xgs, -xgs), axis=0)

    return symmetrized


##################################


######## Losses #######
def sm_sample_loss(
    params: Dict[str, hk.Params],
    x: np.ndarray,  # [1, d]
    g: np.ndarray,  # [1, d]
) -> float:
    """Exact divergence computation."""
    loss = div_v = v_times_s = 0

    for key in params.keys():
        score = score_net.apply(params[key], x, g, key)  # [N, d]
        div_score = div_net.apply(params[key], x, g, key)  # float

        if (cfg.lamb2 + cfg.lamb3 + cfg.lamb4) > 0:
            if key == "x":
                if cfg.lamb2 > 0:
                    div_v += np.sum(div_force(x)) - cfg.eps * div_score

                if cfg.lamb3 + cfg.lamb4 > 0:
                    vx = calc_xdot(x, g) - cfg.eps * score
                    v_times_s += np.sum(vx * score)
            else:
                if cfg.lamb2 > 0:
                    div_v += -cfg.gamma * cfg.d * cfg.N - cfg.gamma * div_score

                if cfg.lamb3 + cfg.lamb4 > 0:
                    vg = -cfg.gamma * g - cfg.gamma * score
                    v_times_s += np.sum(vg * score)

        loss += np.sum(score**2) + 2 * div_score

    return loss, div_v, v_times_s


sm_samples_losses = jax.vmap(sm_sample_loss, in_axes=(None, 0, 0))


def sm_loss(params: hk.Params, xbatch: np.ndarray, gbatch: np.ndarray) -> float:
    losses, div_vs, v_times_s = sm_samples_losses(params, xbatch, gbatch)
    total_loss = 0
    if cfg.lamb1 > 0:
        total_loss += cfg.lamb1 * np.mean(losses) / (2 * cfg.N * cfg.d)
    if cfg.lamb2 > 0:
        total_loss += cfg.lamb2 * np.mean(div_vs) ** 2 / (2 * cfg.N * cfg.d)
    if cfg.lamb3 > 0:
        total_loss += cfg.lamb3 * np.mean(v_times_s) ** 2 / (2 * cfg.N * cfg.d)
    if cfg.lamb4 > 0:
        total_loss += (
            cfg.lamb4 * np.mean((div_vs + v_times_s) ** 2) / (2 * cfg.N * cfg.d)
        )

    return total_loss


def evaluate_target(
    x: np.ndarray, g: np.ndarray, key: str  # [1, d]  # [1, d]
) -> np.ndarray:
    if key == "g":
        return -g
    elif key == "x":
        if cfg.shift_system:
            norm = np.min(
                np.array([np.linalg.norm(x - 2 * cfg.width), np.linalg.norm(x)])
            )
            xhat = -np.sign(x - cfg.width)
        else:
            norm = np.linalg.norm(x)
            xhat = x / norm

        if cfg.smooth_force:
            force = cfg.k * drifts.softplus(2 - norm, cfg.beta) * xhat
        else:
            force = cfg.k * (2 - norm) * xhat * (norm < 2)

        if cfg.clip_force:
            # force = force * (np.linalg.norm(force) > cfg.min_force)
            force = force * (norm < cfg.max_distance)

        return (force - cfg.A * x) / cfg.eps
    else:
        raise ValueError("Key must be x or g")


def supervised_sample_loss(
    params: Dict[str, hk.Params],
    x: np.ndarray,  # [1, d]
    g: np.ndarray,  # [1, d]
) -> float:
    """Exact divergence computation."""
    loss = 0
    for key in params.keys():
        score = score_net.apply(params[key], x, g, key)  # [N, d]
        target = evaluate_target(x, g, key)  # [N, d]
        loss += np.sum((target - score) ** 2)

    if cfg.loss_type == "supervised":
        return loss / (2 * score.size)
    else:
        return loss / np.sum(target**2)


supervised_samples_losses = jax.vmap(supervised_sample_loss, in_axes=(None, 0, 0))
supervised_loss = jax.jit(
    lambda params, xbatch, gbatch: np.mean(
        supervised_samples_losses(params, xbatch, gbatch)
    )
)
#######################


#### Entropy visualization ####
def d_softplus(x: float):
    return 1.0 / (1.0 + np.exp(-x * cfg.beta))


def div_force(x: np.ndarray) -> float:  # [1, d]
    if cfg.shift_system:
        norm = np.min(np.array([np.linalg.norm(x - 2 * cfg.width), np.linalg.norm(x)]))
    else:
        norm = np.linalg.norm(x, axis=1)

    if cfg.smooth_force:
        force = cfg.k * drifts.softplus(2 - norm, cfg.beta)
        force_contribution = force * (cfg.d - 1) / norm - cfg.k * d_softplus(2 - norm)
    else:
        force = cfg.k * (2 - norm)
        force_contribution = (force * (cfg.d - 1) / norm - cfg.k) * (norm < 2)

    if cfg.clip_force:
        # force_contribution = force_contribution * (np.linalg.norm(force) > cfg.min_force)
        force_contribution = force_contribution * (norm < cfg.max_distance)

    return -cfg.A * cfg.d + force_contribution


def calc_divs(
    params: Dict[str, hk.Params], x: np.ndarray, g: np.ndarray  # [1, d]  # [1, d]
) -> Tuple:
    div_sx = particle_div_net.apply(params["x"], x, g, "x")  # [1]
    div_sg = particle_div_net.apply(params["g"], x, g, "g")  # [1]
    div_vx = div_force(x) - cfg.eps * div_sx  # [1]
    div_vg = -cfg.gamma * cfg.d - cfg.gamma * div_sg  # [1]
    div_v = div_vx + div_vg  # [1]

    return div_sx, div_sg, div_vx, div_vg, div_v


def calc_vs(
    xg: np.ndarray,  # [2, d]
    sx: np.ndarray,  # [1, d]
    sg: np.ndarray,  # [1, d]
) -> Tuple:
    x, g = np.split(xg, 2)
    xdot = calc_xdot(x, g) - cfg.eps * sx  # [1, d]
    gdot = -cfg.gamma * g - cfg.gamma * sg  # [1, d]
    v = np.hstack((xdot, gdot))  # [1, 2*d]

    return xdot, gdot, v


def compute_output_info(
    xg: np.ndarray, params: Dict[str, hk.Params]  # [2, d]
) -> Tuple:
    """Compute the entropy, activity, etc."""
    x, g = np.split(xg, 2)  # ([1, d], [1, d])

    sx = score_net.apply(params["x"], x, g, "x")  # [1, d]
    sg = score_net.apply(params["g"], x, g, "g")  # [1, d]
    scores = np.hstack((sx, sg))  # [1, 2*d]
    div_sx, div_sg, div_vx, div_vg, div_v = calc_divs(params, x, g)
    xdot, gdot, v = calc_vs(xg, sx, sg)
    v_times_s = np.sum(v * scores, axis=1)  # [1]
    gdot_mag = np.linalg.norm(gdot, axis=1)  # [1]
    xdot_mag = np.linalg.norm(xdot, axis=1)  # [1]
    score_mag = np.linalg.norm(scores, axis=1)  # [1]
    x_score_mag = np.linalg.norm(sx, axis=1)  # [1]
    g_score_mag = np.linalg.norm(sg, axis=1)  # [1]

    return (
        np.squeeze(x),
        np.squeeze(g),
        np.squeeze(gdot_mag),
        np.squeeze(xdot_mag),
        np.squeeze(score_mag),
        np.squeeze(x_score_mag),
        np.squeeze(g_score_mag),
        np.squeeze(v),
        np.squeeze(v_times_s),
        np.squeeze(div_v),
        np.squeeze(div_vx),
        np.squeeze(div_vg),
        np.squeeze(div_sx),
        np.squeeze(div_sg),
    )


compute_batch_output_info = jax.jit(jax.vmap(compute_output_info, in_axes=(0, None)))


def make_entropy_plot(params: Dict[str, hk.Params]):
    # compute quantities needed for plotting
    inds = onp.random.choice(onp.arange(cfg.ntrajs), size=cfg.plot_bs, replace=False)
    (
        xs,
        gs,
        gdot_mags,
        xdot_mags,
        score_mags,
        x_score_mags,
        g_score_mags,
        vs,
        v_times_s,
        div_vs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    ) = compute_batch_output_info(symmetrize_data(data["xgs"][inds]), params)

    # compute seifert entropy
    xscale = np.array(cfg.d * [cfg.eps])
    gscale = np.array(cfg.d * [cfg.gamma])
    scales = np.concatenate((xscale, gscale))
    seifert_entropy = np.sum(vs / scales[None, :] * vs, axis=1)

    # common plot parameters
    plt.close("all")
    plt.style.use("dark_background")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fraction = 0.15
    shrink = 0.5
    fontsize = 12.5

    ###### main entropy figure
    # individual panels
    titles = [
        [r"$\Vert\dot{g}\Vert$", r"$\Vert\dot{x}\Vert$", r"$\Vert v \Vert_{D^{-1}}^2$"],
        [r"$\Vert s_g\Vert$", r"$\Vert s_x\Vert$", r"$\Vert s \Vert$"],
        [r"$\nabla\cdot v_g$", r"$\nabla\cdot v_x$", r"$\nabla\cdot v$"],
        [r"$\nabla\cdot s_g$", r"$\nabla\cdot s_x$", r"$v \cdot s$"],
    ]

    cs = [
        [gdot_mags, xdot_mags, seifert_entropy],
        [g_score_mags, x_score_mags, score_mags],
        [div_vgs, div_vxs, div_vs],
        [div_sgs, div_sxs, v_times_s],
    ]

    cmaps = [
        sns.color_palette("magma", as_cmap=True),
        sns.color_palette("magma", as_cmap=True),
        sns.color_palette("icefire", as_cmap=True),
        sns.color_palette("icefire", as_cmap=True),
    ]

    nrows = len(titles)
    ncols = len(titles[0])
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    for ax in axs.ravel():
        if cfg.shift_system:
            ax.set_xlim([0, 2 * cfg.width])
        else:
            ax.set_xlim([-cfg.width, cfg.width])

        ax.set_ylim([-3.5, 3.5])
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$g$")

    # do the plotting
    for ii in range(nrows):
        for jj in range(ncols):
            title = titles[ii][jj]
            c = cs[ii][jj]
            ax = axs[ii, jj]
            ax.set_title(title, fontsize=fontsize)

            min_val = float(onp.min(c)) if onp.min(c) < 0 else 0
            max_val = float(onp.max(c))

            # make symmetric
            if ii >= nrows // 2:
                min_val = min(min_val, -max_val)
                max_val = max(max_val, -min_val)

            vmin = min_val
            vmax = max_val

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            scat = ax.scatter(
                xs, gs, s=10, marker="o", c=c, cmap=cmaps[ii], norm=norm, alpha=0.25
            )
            cbar = fig.colorbar(scat, ax=ax, fraction=fraction, shrink=shrink)
            cbar.ax.tick_params(labelsize=fontsize)

            # fix alpha on colorbar
            cbar.set_alpha(1.0)
            cbar.draw_all()

    wandb.log({"entropy_figure": wandb.Image(fig)})


def compute_sample_convergence_statistics(
    params: Dict[str, hk.Params], xg: np.ndarray  # [2, d]
) -> None:
    x, g = np.split(xg, 2)
    div_v = calc_divs(params, x, g)[4]  # [1]
    sx = score_net.apply(params["x"], x, g, "x")  # [1, d]
    sg = score_net.apply(params["g"], x, g, "g")  # [1, d]
    v = calc_vs(xg, sx, sg)[2]  # [1, 2*d]

    scores = np.hstack((sx, sg))  # [1, 2*d]
    v_times_s = np.sum(v * scores, axis=1)  # [1]

    return div_v, v_times_s, (div_v + v_times_s) ** 2


batch_convergence_statistics = jax.vmap(
    compute_sample_convergence_statistics, in_axes=(None, 0)
)


@jax.jit
def compute_convergence_statistics(
    params: Dict[str, hk.Params],
    xbatch: np.ndarray,  # [bs, 1, d]
    gbatch: np.ndarray,  # [bs, 1, d]
) -> Tuple:
    xg_batch = np.concatenate((xbatch, gbatch), axis=1)  # [bs, 2, d]
    batch_div_vs, batch_v_times_s, batch_pinn = batch_convergence_statistics(
        params, xg_batch
    )
    return np.mean(batch_div_vs), np.mean(batch_v_times_s), np.mean(batch_pinn)


##############################


def step_data(xgs: onp.ndarray, prng_key: np.ndarray) -> Tuple[onp.ndarray, np.ndarray]:
    for curr_batch in range(cfg.nbatches_online):
        lb = cfg.bs_online * curr_batch
        ub = lb + cfg.bs_online
        batch_xgs = xgs[lb:ub]
        noises = jax.random.normal(
            prng_key, shape=(batch_xgs.shape[0], cfg.nsteps_online, 2 * cfg.N, cfg.d)
        )
        xgs[lb:ub] = rollout_trajs(batch_xgs, noises)
        prng_key = jax.random.split(prng_key)[0]

    return symmetrize_data(xgs), prng_key


def get_batches(
    curr_batch: int,
    xs: np.ndarray,
    gs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lb = cfg.bs * curr_batch
    ub = lb + cfg.bs
    xbatch, gbatch = xs[lb:ub], gs[lb:ub]
    xbatch = xbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))
    gbatch = gbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))

    return (xbatch, gbatch)


def log_metrics(
    curr_batch: int,
    curr_epoch: int,
    xbatch: np.ndarray,
    gbatch: np.ndarray,
    loss_value: np.ndarray,
    grads: Dict[str, hk.Params],
    validation_params: Dict[str, hk.Params],
    ema_params: Dict[str, hk.Params],
    curr_params: Dict[str, hk.Params],
) -> None:
    ## log loss and score norm
    score_norm = 0
    xbatch = xbatch.reshape((-1, cfg.N, cfg.d))
    gbatch = gbatch.reshape((-1, cfg.N, cfg.d))
    for key in validation_params.keys():
        scores = map_score_net(
            validation_params[key], xbatch, gbatch, key
        )  # [bs, 1, d]
        score_norm += np.mean(np.sum(scores**2, axis=(1, 2))) / (2 * cfg.N * cfg.d)

    if cfg.loss_type == "score_matching":
        supervised_loss_val = supervised_loss(validation_params, xbatch, gbatch)
    else:
        supervised_loss_val = None

    wandb.log(
        {
            f"loss": loss_value[0],
            f"score_norm": -score_norm,
            f"grad": losses.compute_grad_norm(unreplicate(grads)),
            f"supervised_loss": supervised_loss_val,
        }
    )

    if (curr_batch + curr_epoch * cfg.nbatches) % cfg.stat_freq == 0:
        div_v, v_times_s, pinn = compute_convergence_statistics(
            validation_params, xbatch, gbatch
        )
        wandb.log({"div_v": div_v, "v_times_s": v_times_s, "pinn": pinn})

    if (curr_batch + curr_epoch * cfg.nbatches) % cfg.visual_freq == 0:
        make_entropy_plot(validation_params)

    if (curr_batch + curr_epoch * cfg.nbatches) % cfg.save_freq == 0:
        data["params_list"].append(curr_params)
        data["ema_params_list"].append(ema_params)
        pickle.dump(data, open(f"{cfg.output_folder}/{cfg.output_name}.npy", "wb"))


def train_loop(
    prng_key: np.ndarray, opt: optax.GradientTransformation, opt_state: optax.OptState
) -> None:
    """Carry out the training loop."""
    ## set up data and output
    loss = setup_loss()
    params = replicate(data["params_list"][-1])
    ema_params = {
        ema_fac: deepcopy(data["params_list"][-1]) for ema_fac in cfg.ema_facs
    }
    xgs = data["xgs"]

    ## perform training
    start_time = time.time()
    for curr_epoch in tqdm(range(cfg.n_epochs)):
        xgs, prng_key = step_data(xgs[: cfg.ntrajs], prng_key)
        xs, gs = np.split(xgs, 2, axis=1)  # [2*ntrajs, 1, d], [2*ntrajs, 1, d]

        for curr_batch in tqdm(range(cfg.nbatches)):
            xbatch, gbatch = get_batches(curr_batch, xs, gs)

            ## set up loss function arguments
            if cfg.loss_type == "score_matching":
                loss_args = (xbatch, gbatch)
            elif cfg.loss_type == "supervised":
                loss_args = (xbatch, gbatch)

            ## perform update
            params, opt_state, loss_value, grads = losses.pupdate(
                params, opt_state, opt, loss, loss_args
            )

            ## compute EMA params
            curr_params = unreplicate(params)
            ema_params = update_ema_params(curr_params, ema_params)
            # validation_params = ema_params[cfg.ema_facs[-1]]
            validation_params = curr_params

            log_metrics(
                curr_batch,
                curr_epoch,
                xbatch,
                gbatch,
                loss_value,
                grads,
                validation_params,
                ema_params,
                curr_params,
            )

    # dump one final time
    pickle.dump(data, open(f"{cfg.output_folder}/{cfg.output_name}.npy", "wb"))


#### Initialization ####
def setup_loss() -> Callable:
    """Set up the loss function."""
    if cfg.loss_type == "score_matching":
        return sm_loss
    elif cfg.loss_type == "supervised":
        return supervised_loss
    else:
        raise ValueError("Loss type not implemented")


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
        description="Run a one-d mips simulation from the command line."
    )
    parser.add_argument("--ntrajs", type=int)
    parser.add_argument("--nsteps_online", type=int)
    parser.add_argument("--network_path", type=str)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--bs_online", type=int)
    parser.add_argument("--plot_bs", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dt_online", type=float)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--width", type=float)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--lamb1", type=float)
    parser.add_argument("--lamb2", type=float)
    parser.add_argument("--lamb3", type=float)
    parser.add_argument("--lamb4", type=float)
    parser.add_argument("--shift_network", type=int)
    parser.add_argument("--shift_system", type=int)
    parser.add_argument("--symmetric_network", type=int)
    parser.add_argument("--clip_force", type=int)
    parser.add_argument("--smooth_force", type=int)
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--n_neurons", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--decay_steps", type=int)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)

    return parser.parse_args()


def setup_config_dict():
    cfg = config_dict.ConfigDict()
    cfg.d = 1
    cfg.r = 1.0
    cfg.N = 1
    cfg.clip = 1.0
    cfg.thermalize_fac = 10.0

    cfg.max_n_steps = 50
    cfg.n_epochs = int(1e6)
    cfg.save_freq = int(1e4)
    cfg.visual_freq = int(5e2)
    cfg.stat_freq = int(1e2)

    ## input parameters
    args = get_simulation_parameters()
    cfg.ntrajs = args.ntrajs
    cfg.nsteps_online = args.nsteps_online
    cfg.bs = args.bs
    cfg.bs_online = args.bs_online
    cfg.plot_bs = args.plot_bs
    cfg.dt = args.dt
    cfg.dt_online = args.dt_online
    cfg.w0 = args.w0
    cfg.v0 = args.v0
    cfg.A = args.A
    cfg.k = args.k
    cfg.beta = args.beta
    cfg.width = args.width
    cfg.gamma = args.gamma
    cfg.eps = args.eps
    cfg.loss_type = args.loss_type
    cfg.lamb1 = args.lamb1
    cfg.lamb2 = args.lamb2
    cfg.lamb3 = args.lamb3
    cfg.lamb4 = args.lamb4
    cfg.shift_network = args.shift_network
    cfg.shift_system = args.shift_system
    cfg.symmetric_network = args.symmetric_network
    cfg.clip_force = args.clip_force
    cfg.smooth_force = args.smooth_force
    cfg.max_distance = cfg.width / 2
    cfg.min_force = 1e-8
    cfg.n_hidden = args.n_hidden
    cfg.n_neurons = args.n_neurons
    cfg.ema_facs = [0.999, 0.9999]
    cfg.learning_rate = args.learning_rate
    cfg.decay_steps = args.decay_steps
    cfg.wandb_name = f"{args.wandb_name}_{args.slurm_id}"
    cfg.output_name = f"{args.output_name}_{args.slurm_id}"
    cfg.output_folder = args.output_folder

    if cfg.gamma == 0:
        cfg.tburn = cfg.thermalize_fac / cfg.eps
    elif cfg.eps == 0:
        cfg.tburn = cfg.thermalize_fac / cfg.gamma
    else:
        cfg.tburn = cfg.thermalize_fac / min(cfg.gamma, cfg.eps)

    cfg.nbatches = int(2 * cfg.ntrajs / cfg.bs)
    cfg.nbatches_online = (
        int(cfg.nbatches / cfg.bs_online) // 2
    )  # only move un-symmetrized data
    cfg.nbatches += 1 if cfg.nbatches * cfg.bs < cfg.ntrajs else 0
    cfg.nbatches_online += 1 if cfg.nbatches_online * cfg.bs_online < cfg.ntrajs else 0
    cfg.nsteps = int(cfg.tburn / cfg.dt + 1)
    cfg.ndevices = jax.local_device_count()

    ## based on input parameters
    cfg.dim = 2 * cfg.N * cfg.d
    cfg.radii = onp.ones(cfg.N) * cfg.r
    cfg.sig0x, cfg.sig0g = cfg.width / 2, 1.0

    return cfg, args


def construct_network():
    shift_func, particle_div_shift_func, div_shift_func = define_shift_functions()
    score_net, particle_div_net, div_net = networks.define_full_particle_split_mlp(
        cfg.w0,
        cfg.d,
        cfg.N,
        cfg.n_hidden,
        cfg.n_neurons,
        shift_func,
        particle_div_shift_func,
        div_shift_func,
        symmetric=cfg.symmetric_network,
        symmetric_point=(
            np.array([2 * cfg.width, 0.0]) if cfg.shift_system else np.zeros(2 * cfg.d)
        ),
    )
    map_score_net = jax.jit(
        jax.vmap(score_net.apply, in_axes=(None, 0, 0, None)), static_argnums=3
    )

    return score_net, particle_div_net, div_net, map_score_net


def initialize_network(prng_key: np.ndarray):
    if args.network_path != "":
        params = pickle.load(open(args.network_path, "rb"))["params_list"][-1]
    else:
        ex_xs, ex_gs = np.split(xgs[0], 2)
        key1, key2 = jax.random.split(prng_key)
        params = {
            "x": score_net.init(key1, ex_xs, ex_gs, "x"),
            "g": score_net.init(key2, ex_xs, ex_gs, "g"),
        }
        prng_key = jax.random.split(key1)[0]
        print(
            f"Number of parameters: {2*jax.flatten_util.ravel_pytree(params['x'])[0].size}"
        )

    return params, prng_key


#####################


#### Helper Functions ######
@jax.jit
def update_ema_params(
    curr_params: hk.Params, ema_params: Dict[float, Dict[str, hk.Params]]
) -> Dict[float, hk.Params]:
    new_ema_params = {}
    for ema_fac in cfg.ema_facs:
        curr_ema_params = {}
        for key, params in curr_params.items():
            curr_ema_params[key] = jax.tree_util.tree_map(
                lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
                params,
                ema_params[ema_fac][key],
            )
        new_ema_params[ema_fac] = curr_ema_params

    return new_ema_params


def define_shift_functions() -> Tuple[Callable, Callable]:
    if cfg.shift_network:

        def shift_func(
            x: np.ndarray, g: np.ndarray, key: str  # [1, d]  # [1, d]
        ) -> np.ndarray:
            if key == "x":
                return calc_xdot(x, g) / cfg.eps
            elif key == "g":
                return -g
            else:
                raise ValueError("Key must be x or g.")

        def particle_div_shift_func(
            x: np.ndarray, g: np.ndarray, key: str  # [1, d]  # [1, d]
        ) -> np.ndarray:
            if key == "x":
                return div_force(x) / cfg.eps
            elif key == "g":
                return -cfg.d * np.ones(cfg.N)
            else:
                raise ValueError("Key must be x or g.")

        div_shift_func = lambda xs, gs, key: np.sum(
            particle_div_shift_func(xs, gs, key)
        )
    else:
        shift_func = particle_div_shift_func = div_shift_func = lambda xs, gs, key: 0.0

    return shift_func, particle_div_shift_func, div_shift_func


#######################

if __name__ == "__main__":
    cfg, args = setup_config_dict()
    prng_key = jax.random.PRNGKey(onp.random.randint(1000))
    xgs, prng_key = generate_data(prng_key)
    score_net, particle_div_net, div_net, map_score_net = construct_network()
    params, prng_key = initialize_network(prng_key)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=0,
        decay_steps=cfg.decay_steps,
    )

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip), optax.adam(learning_rate=schedule)
    )

    ## for parallel training
    opt_state = replicate(opt.init(params))

    ## set up weights and biases tracking
    wandb.init(
        project="",
        name=cfg.wandb_name,
        config=cfg.to_dict(),
    )

    ## train the model
    data = {"params_list": [params], "ema_params_list": [], "xgs": xgs, "cfg": cfg}

    train_loop(prng_key, opt, opt_state)
