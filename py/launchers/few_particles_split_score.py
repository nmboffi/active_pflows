"""
Nicholas M. Boffi
6/18/23

Code for systematic simulations of the Cates system with few particles.
Split the learning of the score into separate x and g scores.
"""

import jax
import jax.numpy as np
import numpy as onp
import dill as pickle
from typing import Tuple, Callable, Dict, Union
from ml_collections import config_dict
from copy import deepcopy
import argparse


import time
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import functools
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


###### sensible matplotlib defaults ######
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
#########################################


########## Dataset ###############
@functools.partial(jax.jit, static_argnums=2)
def rollout(
    init_xg: np.ndarray,  # [2N*d]
    noises: np.ndarray,  # [nsteps, 2*N, d]
    cfg: config_dict.FrozenConfigDict,
) -> np.ndarray:
    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = step_system(xg, noise, cfg)
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final, xg_traj


@functools.partial(jax.jit, static_argnums=2)
def rollout_trajs(
    init_xgs: np.ndarray,  # [ntrajs, 2*N*d]
    noises: np.ndarray,  # [ntrajs, nsteps, 2*N, d]
    cfg: config_dict.FrozenConfigDict,
) -> np.ndarray:
    return jax.vmap(lambda init_xg, traj_noises: rollout(cfg, init_xg, traj_noises)[0])(
        init_xgs, noises
    )


def generate_data(
    cfg: config_dict.ConfigDict, key: np.ndarray, load_data: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset from the example problem."""
    if load_data:
        # load in the data
        cfg.load_data = True
        cfg.data_folder = args.data_folder

        try:
            data_name = (
                f"v0={cfg.v0}_gamma={cfg.gamma}_eps={cfg.eps}_phi={cfg.phi}"
                + f"_dt={cfg.dt}_beta={cfg.beta}_A={cfg.A}_k={cfg.k}_N={cfg.N}"
            )
            cfg.data_path = f"{cfg.data_folder}/{data_name}.npy"
            xgs = pickle.load(open(cfg.data_path, "rb"))["xgs"]
        except:
            # backwards compatibility before saving value of k
            data_name = (
                f"v0={cfg.v0}_gamma={cfg.gamma}_eps={cfg.eps}_phi={cfg.phi}"
                + f"_dt={cfg.dt}_beta={cfg.beta}_A={cfg.A}_N={cfg.N}"
            )
            cfg.data_path = f"{cfg.data_folder}/{data_name}.npy"
            xgs = pickle.load(open(cfg.data_path, "rb"))["xgs"]

        # update ntrajs and batch size accordingly
        cfg.ntrajs = xgs.shape[0]
        cfg.nbatches = int(cfg.ntrajs / cfg.bs)
        cfg.nbatches += 1 if cfg.nbatches * cfg.bs < cfg.ntrajs else 0

        return onp.array(xgs), key, config_dict.FrozenConfigDict(cfg)
    else:
        xs = drifts.torus_project(
            cfg.sig0x * onp.random.randn(cfg.ntrajs, cfg.N, cfg.d), cfg.width
        )
        gs = cfg.sig0g * onp.random.randn(cfg.ntrajs, cfg.N, cfg.d)
        xgs = onp.concatenate((xs, gs), axis=1)
        nbatches_rollout = int(cfg.nsteps / cfg.max_n_steps) + 1
        cfg = config_dict.FrozenConfigDict(cfg)

        start_time = time.time()
        print(f"Starting data generation.")
        for curr_batch in range(nbatches_rollout):
            noises = jax.random.normal(
                key, shape=(cfg.ntrajs, cfg.max_n_steps, 2 * cfg.N, cfg.d)
            )
            xgs = rollout_trajs(xgs, noises, cfg)
            key = jax.random.split(key)[0]
        end_time = time.time()
        print(f"Finished data generation. Total time={(end_time-start_time)/60.}m")

        return onp.array(xgs), key, cfg


##################################


######## Losses #######
def sm_sample_loss(
    params: Dict[str, hk.Params],
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    noises: Union[None, np.ndarray],  # [2N, d]
    cfg: config_dict.ConfigDict,
) -> float:
    """Exact divergence computation."""
    loss = div_v = v_times_s = 0
    if noises != None:
        noise_x, noise_g = np.split(noises, 2)

    for key in params.keys():
        ## compute score, either exactly or via skilling-hutchinson
        if cfg.use_skilling_hutch:
            if key == "x":
                jvp_fun = lambda xs: score_net.apply(params[key], xs, gs, key)
                primals = (xs,)
                tangents = (noise_x,)
            elif key == "g":
                jvp_fun = lambda gs: score_net.apply(params[key], xs, gs, key)
                primals = (gs,)
                tangents = (noise_g,)

            score, jvp = jax.jvp(jvp_fun, primals, tangents)
            div_score = np.sum(tangents[0] * jvp)
        else:
            score = score_net.apply(params[key], xs, gs, key)  # [N, d]
            div_score = div_net.apply(params[key], xs, gs, key)  # float

        if (cfg.lamb2 + cfg.lamb3 + cfg.lamb4) > 0:
            if key == "x":
                if cfg.lamb2 > 0:
                    div_v += np.sum(div_force(xs, cfg)) - cfg.eps * div_score

                if cfg.lamb3 + cfg.lamb4 > 0:
                    vx = calc_xdots(np.concatenate((xs, gs)), cfg) - cfg.eps * score
                    v_times_s += np.sum(vx * score)
            else:
                if cfg.lamb2 > 0:
                    div_v += -cfg.gamma * cfg.d * cfg.N - cfg.gamma * div_score

                if cfg.lamb3 + cfg.lamb4 > 0:
                    vg = -cfg.gamma * gs - cfg.gamma * score
                    v_times_s += np.sum(vg * score)

        loss += np.sum(score**2) + 2 * div_score

    return loss, div_v, v_times_s


sm_samples_losses = jax.vmap(sm_sample_loss, in_axes=(None, 0, 0, 0, None))


def sm_loss(
    params: hk.Params,
    xbatch: np.ndarray,
    gbatch: np.ndarray,
    noise_batch: np.ndarray,
    cfg: config_dict.ConfigDict,
) -> float:
    losses, div_vs, v_times_s = sm_samples_losses(
        params, xbatch, gbatch, noise_batch, cfg
    )
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


def denoising_sample_loss(
    params: Dict[str, hk.Params],
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    noises: np.ndarray,  # [2*N, d]
    cfg: config_dict.ConfigDict,
) -> float:
    """Exact denoising loss, up to discretization errors."""
    xgs = np.concatenate((xs, gs))
    loss = div_v = v_times_s = 0

    ## antithetic sampling.
    for flip in range(2):
        ## take a forward step using +- the noise.
        noises = noises if flip == 0 else -noises
        noise_x, noise_g = np.split(noises, 2)
        xgs_next = step_system(xgs, noises, cfg)
        xs_next, gs_next = np.split(xgs_next, 2)

        for key in params.keys():
            ## set up the target and the scale.
            if key == "x":
                scale = np.sqrt(2 * cfg.dt_online * cfg.eps)
                target_noise = noise_x
            elif key == "g":
                scale = np.sqrt(2 * cfg.dt_online * cfg.gamma)
                target_noise = noise_g

            ## compute the score, the divergence, and the loss.
            score = score_net.apply(params[key], xs_next, gs_next, key)  # [N, d]
            div_score = np.sum(score * target_noise) / scale
            loss += 0.5 * (np.sum(score**2) + 2 * div_score)

            ## compute (optional) regularization terms.
            ## note that there is an extra factor of 0.5 in the code below; this is
            ## because we are using antithetic sampling.
            ## we evaluate the divergences at the current time step, while we evaluate
            ## the velocities at the next time step. this is self-consistent
            ## with the approximation of the divergence above.
            if (cfg.lamb2 + cfg.lamb3 + cfg.lamb4) > 0:
                if key == "x":
                    if cfg.lamb2 > 0:
                        div_v += 0.5 * (
                            np.sum(div_force(xs, cfg)) - cfg.eps * div_score
                        )

                    if cfg.lamb3 + cfg.lamb4 > 0:
                        vx = calc_xdots(xgs_next, cfg) - cfg.eps * score
                        v_times_s += 0.5 * np.sum(vx * score)
                else:
                    if cfg.lamb2 > 0:
                        div_v += 0.5 * (
                            -cfg.gamma * cfg.d * cfg.N - cfg.gamma * div_score
                        )

                    if cfg.lamb3 + cfg.lamb4 > 0:
                        vg = -cfg.gamma * gs_next - cfg.gamma * score
                        v_times_s += 0.5 * np.sum(vg * score)

    return loss, div_v, v_times_s


denoising_samples_losses = jax.vmap(
    denoising_sample_loss, in_axes=(None, 0, 0, 0, None)
)


def denoising_loss(
    params: hk.Params,
    xbatch: np.ndarray,
    gbatch: np.ndarray,
    noise_batch: np.ndarray,
    cfg: config_dict.ConfigDict,
) -> float:
    losses, div_vs, v_times_s = denoising_samples_losses(
        params, xbatch, gbatch, noise_batch, cfg
    )
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
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    key: str,
    cfg: config_dict.ConfigDict,
) -> np.ndarray:
    if key == "g":
        return -gs
    elif key == "x":
        return (
            cfg.k
            * np.sum(
                drifts.elastic_interaction(
                    xs, np.ones(cfg.N), cfg.N, cfg.width, cfg.beta
                ),
                axis=1,
            )
            - cfg.A * xs
        ) / cfg.eps
    else:
        raise ValueError("Key must be x or g")


def supervised_sample_loss(
    params: Dict[str, hk.Params],
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    cfg: config_dict.ConfigDict,
) -> float:
    """Exact divergence computation."""
    loss = 0
    for key in params.keys():
        score = score_net.apply(params[key], xs, gs, key)  # [N, d]
        target = evaluate_target(xs, gs, key, cfg)  # [N, d]
        loss += np.sum((target - score) ** 2)

    if cfg.loss_type == "supervised":
        return loss / (2 * score.size)
    else:
        return loss / np.sum(target**2)


supervised_samples_losses = jax.vmap(supervised_sample_loss, in_axes=(None, 0, 0, None))

supervised_loss = jax.jit(
    lambda params, xbatch, gbatch, cfg: np.mean(
        supervised_samples_losses(params, xbatch, gbatch, cfg)
    ),
    static_argnums=3,
)
#######################


### Entropy visualization ####
def d_softplus(x: float, cfg: config_dict.ConfigDict):
    return 1.0 / (1.0 + np.exp(-x * cfg.beta))


def div_particle_force(
    xs: np.ndarray,  # [N, d]
    ii: int,
    cfg: config_dict.ConfigDict,
) -> float:
    r"""Compute \nabla_i \cdot b_i."""
    xi = xs[ii]
    diffs = jax.vmap(lambda xj: drifts.wrapped_diff(xi, xj, cfg.width))(xs)
    rijs = np.linalg.norm(diffs, axis=1)
    rslts = cfg.k * (
        drifts.softplus(2 * cfg.r - rijs, cfg.beta) * (cfg.d - 1) / rijs
        - d_softplus(2 * cfg.r - rijs, cfg)
    )
    rslts = rslts.at[ii].set(0.0)  # fix divide-by-zero
    return np.sum(rslts) - cfg.A * cfg.d


calc_xdots = lambda xgs, cfg: drifts.mips(
    xgs, cfg.v0, cfg.A, cfg.k, cfg.gamma, np.ones(cfg.N), cfg.width, cfg.beta, cfg.N
)[: cfg.N]


step_system = lambda xgs, noise, cfg: drifts.step_mips_OU_EM(
    xgs,
    cfg.dt_online,
    np.ones(cfg.N),
    cfg.A,
    cfg.k,
    cfg.v0,
    cfg.N,
    cfg.d,
    cfg.eps,
    cfg.gamma,
    cfg.width,
    cfg.beta,
    noise,
)


div_force = lambda xs, cfg: jax.vmap(div_particle_force, in_axes=(None, 0, None))(
    xs, np.arange(cfg.N), cfg
)


def calc_divs(
    params: Dict[str, hk.Params],
    xs: np.ndarray,  # [N, d]
    gs: np.ndarray,  # [N, d]
    cfg: config_dict.ConfigDict,
    particle_div_net: Callable,
) -> Tuple:  # [N]
    div_sxs = particle_div_net.apply(params["x"], xs, gs, "x")  # [N]
    div_sgs = particle_div_net.apply(params["g"], xs, gs, "g")  # [N]
    div_vxs = div_force(xs, cfg) - cfg.eps * div_sxs  # [N]
    div_vgs = -cfg.gamma * cfg.d - cfg.gamma * div_sgs  # [N]
    div_vs = div_vxs + div_vgs  # [N]

    return div_sxs, div_sgs, div_vxs, div_vgs, div_vs


def calc_vs(
    xgs: np.ndarray,  # [2*N, d]
    sxs: np.ndarray,  # [N, d]
    sgs: np.ndarray,  # [N, d]
    cfg: config_dict.ConfigDict,
) -> Tuple:
    gs = xgs[cfg.N :]
    xdots = calc_xdots(xgs, cfg) - cfg.eps * sxs  # [N, d]
    gdots = -cfg.gamma * gs - cfg.gamma * sgs  # [N, d]
    vs = np.hstack((xdots, gdots))  # [N, 2*d]

    return xdots, gdots, vs


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def compute_output_info(
    xgs: np.ndarray,
    params: Dict[str, hk.Params],
    cfg: config_dict.ConfigDict,
    score_net: Callable,
    particle_div_net: Callable,
) -> Tuple:
    """Compute the entropy, activity, etc."""
    xs, gs = np.split(xgs, 2)  # ([N, d], [N, d])
    sxs = score_net.apply(params["x"], xs, gs, "x")  # [N, d]
    sgs = score_net.apply(params["g"], xs, gs, "g")  # [N, d]
    particle_scores = np.hstack((sxs, sgs))  # [N, 2*d]
    div_sxs, div_sgs, div_vxs, div_vgs, div_vs = calc_divs(
        params, xs, gs, cfg, particle_div_net
    )
    xdots, gdots, vs = calc_vs(xgs, sxs, sgs, cfg)
    v_times_s = np.sum(vs * particle_scores, axis=1)  # [N]
    gdot_mags = np.linalg.norm(gdots, axis=1)  # [N]
    xdot_mags = np.linalg.norm(xdots, axis=1)  # [N]
    particle_score_mags = np.linalg.norm(particle_scores, axis=1)  # [N]
    x_score_mags = np.linalg.norm(sxs, axis=1)  # [N]
    g_score_mags = np.linalg.norm(sgs, axis=1)  # [N]

    return (
        xs,
        gs,
        gdot_mags,
        xdot_mags,
        particle_score_mags,
        x_score_mags,
        g_score_mags,
        vs,
        v_times_s,
        div_vs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    )


def make_entropy_plot(
    params: Dict[str, hk.Params],
    cfg: config_dict.ConfigDict,
) -> None:
    # compute quantities needed for plotting
    ind = onp.random.randint(cfg.ntrajs)
    (
        xs,
        gs,
        gdot_mags,
        xdot_mags,
        particle_score_mags,
        x_score_mags,
        g_score_mags,
        vs,
        v_times_s,
        div_vs,
        div_vxs,
        div_vgs,
        div_sxs,
        div_sgs,
    ) = compute_output_info(data["xgs"][ind], params, cfg)

    # compute seifert entropy
    scales = np.array([cfg.eps, cfg.eps, cfg.gamma, cfg.gamma])
    seifert_entropy = np.sum(vs / scales[None, :] * vs, axis=1)

    # common plot parameters
    plt.close("all")
    plt.style.use("dark_background")
    sns.set_palette("deep")
    fw, fh = 4, 4

    if cfg.N > 64:
        scale_fac = 0.35
    else:
        scale_fac = 0.5

    fraction = 0.15
    shrink = 0.5
    fontsize = 12.5

    ###### main entropy figure
    # individual panels
    titles = [
        [
            r"$\Vert\dot{g}\Vert$",
            r"$\Vert\dot{x}\Vert$",
            r"$\Vert v \Vert_{D^{-1}}^2$",
        ],
        [r"$\Vert s_g\Vert$", r"$\Vert s_x\Vert$", r"$\Vert s \Vert$"],
        [r"$\nabla\cdot v_g$", r"$\nabla\cdot v_x$", r"$\nabla\cdot v$"],
        [r"$\nabla\cdot s_g$", r"$\nabla\cdot s_x$", r"$v \cdot s$"],
    ]

    cs = [
        [gdot_mags, xdot_mags, seifert_entropy],
        [g_score_mags, x_score_mags, particle_score_mags],
        [div_vgs, div_vxs, div_vs],
        [div_sgs, div_sxs, v_times_s],
    ]

    cmaps = [
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
        sns.color_palette("mako", as_cmap=True),
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
    axs = axs.reshape((nrows, ncols))

    for ax in axs.ravel():
        ax.set_xlim([-cfg.width, cfg.width])
        ax.set_ylim([-cfg.width, cfg.width])
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.axes.set_aspect(1.0)
        scale = ax.transData.get_matrix()[0, 0]
        ax.tick_params(axis="both", labelsize=fontsize)

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
            #            if ii >= nrows // 2:
            #                min_val = min(min_val, -max_val)
            #                max_val = max(max_val, -min_val)

            vmin = min_val
            vmax = max_val

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            scat = ax.scatter(
                xs[:, 0],
                xs[:, 1],
                s=(scale * scale_fac * cfg.r) ** 2,
                marker="o",
                c=c,
                cmap=cmaps[ii],
                norm=norm,
            )
            cbar = fig.colorbar(scat, ax=ax, fraction=fraction, shrink=shrink)
            cbar.ax.tick_params(labelsize=fontsize)

    wandb.log({"entropy_figure": wandb.Image(fig)})


def compute_sample_convergence_statistics(
    params: Dict[str, hk.Params],
    xgs: np.ndarray,  # [2*N, d]
    cfg: config_dict.ConfigDict,
) -> None:
    xs, gs = np.split(xgs, 2)
    div_vs = calc_divs(params, xs, gs, cfg, particle_div_net)[4]  # [N]
    sxs = score_net.apply(params["x"], xs, gs, "x")  # [N, d]
    sgs = score_net.apply(params["g"], xs, gs, "g")  # [N, d]
    vs = calc_vs(xgs, sxs, sgs, cfg)[2]  # [N, 2*d]
    particle_scores = np.hstack((sxs, sgs))  # [N, 2*d]
    v_times_s = np.sum(vs * particle_scores, axis=1)  # [N]

    return np.sum(div_vs), np.sum(v_times_s), np.sum(div_vs + v_times_s) ** 2


batch_convergence_statistics = jax.vmap(
    compute_sample_convergence_statistics, in_axes=(None, 0, None)
)


@functools.partial(jax.jit, static_argnums=(3,))
def compute_convergence_statistics(
    params: Dict[str, hk.Params],
    xbatch: np.ndarray,  # [bs, N, d]
    gbatch: np.ndarray,  # [bs, N, d]
    cfg: config_dict.ConfigDict,
) -> Tuple:
    xg_batch = np.concatenate((xbatch, gbatch), axis=1)  # [bs, 2*N, d]
    batch_div_vs, batch_v_times_s, batch_discrep = batch_convergence_statistics(
        params, xg_batch, cfg
    )
    return np.sum(batch_div_vs), np.sum(batch_v_times_s), np.sum(batch_discrep)


##############################


def step_data(
    xgs: onp.ndarray,  # [ntrajs, 2N, d]
    prng_key: np.ndarray,
    cfg: config_dict.ConfigDict,
) -> Tuple[np.ndarray, np.ndarray]:
    for curr_batch in range(cfg.nbatches_online):
        lb = cfg.bs_online * curr_batch
        ub = lb + cfg.bs_online
        batch_xgs = xgs[lb:ub]
        noises = jax.random.normal(
            prng_key, shape=(batch_xgs.shape[0], cfg.nsteps_online, 2 * cfg.N, cfg.d)
        )
        xgs[lb:ub] = rollout_trajs(batch_xgs, noises)
        prng_key = jax.random.split(prng_key)[0]

    return xgs, prng_key


def setup_loss_fn_args(
    xs: np.ndarray,  # [ntrajs, N, d]
    gs: np.ndarray,  # [ntrajs, N, d]
    prng_key: np.ndarray,
    curr_batch: int,
    cfg: config_dict.ConfigDict,
) -> Tuple:
    lb = cfg.bs * curr_batch
    ub = lb + cfg.bs
    xbatch, gbatch = xs[lb:ub], gs[lb:ub]
    xbatch = xbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))
    gbatch = gbatch.reshape((cfg.ndevices, -1, cfg.N, cfg.d))
    loss_fn_args = (xbatch, gbatch)

    if "denoising" in cfg.loss_type:
        noise_batch = jax.random.normal(
            prng_key, shape=(xbatch.shape[0], xbatch.shape[1], 2 * cfg.N, cfg.d)
        )
        key = jax.random.split(prng_key)[0]
        loss_fn_args += (noise_batch,)
    elif cfg.loss_type == "score_matching":
        loss_fn_args += (None,)

    return loss_fn_args, prng_key


def log_metrics(
    data: dict,
    curr_params: Dict[str, hk.Params],
    ema_params: Dict[str, hk.Params],
    xbatch: onp.ndarray,
    gbatch: onp.ndarray,
    grads: Dict[str, hk.Params],
    loss_value: float,
    curr_batch: int,
    curr_epoch: int,
    cfg: config_dict.ConfigDict,
) -> None:
    score_norm = 0
    xbatch = xbatch.reshape((-1, cfg.N, cfg.d))
    gbatch = gbatch.reshape((-1, cfg.N, cfg.d))
    for key in curr_params.keys():
        scores = map_score_net(curr_params[key], xbatch, gbatch, key)  # [bs, N, d]
        score_norm += np.mean(np.sum(scores**2, axis=(1, 2))) / (2 * cfg.N * cfg.d)

    supervised_loss_val = supervised_loss(curr_params, xbatch, gbatch, cfg)

    wandb.log(
        {
            f"loss": loss_value,
            f"score_norm": -score_norm,
            f"grad": losses.compute_grad_norm(unreplicate(grads)),
            f"supervised_loss": supervised_loss_val,
        }
    )

    iteration = curr_batch + curr_epoch * cfg.nbatches
    if (iteration % cfg.stat_freq) == 0:
        div_v, v_times_s, pinn = 0, 0, 0
        for curr_batch in range(cfg.nbatches_stats):
            lb = curr_batch * cfg.bs_stats
            ub = lb + cfg.bs_stats
            curr_div_v, curr_v_times_s, curr_pinn = compute_convergence_statistics(
                curr_params, xbatch[lb:ub], gbatch[lb:ub], cfg
            )

            div_v += curr_div_v / cfg.bs
            v_times_s += curr_v_times_s / cfg.bs
            pinn += curr_pinn / cfg.bs

        wandb.log({"div_v": div_v, "v_times_s": v_times_s, "pinn": pinn})

    if (iteration % cfg.visual_freq) == 0:
        make_entropy_plot(curr_params, cfg)

    if (iteration % cfg.save_freq) == 0:
        data["params"] = jax.device_put(curr_params, jax.devices("cpu")[0])
        data["ema_params"] = jax.device_put(ema_params, jax.devices("cpu")[0])
        pickle.dump(
            data,
            open(
                f"{cfg.output_folder}/{cfg.output_name}_{iteration//cfg.save_freq}.npy",
                "wb",
            ),
        )

    return data


def train_loop(
    prng_key: np.ndarray,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    data: dict,
    cfg: config_dict.ConfigDict,
) -> None:
    """Carry out the training loop."""
    ## set up data and output
    loss = setup_loss(cfg)
    params = replicate(data["params"])
    ema_params = {ema_fac: deepcopy(data["params"]) for ema_fac in cfg.ema_facs}
    xgs = data["xgs"]

    ## perform training
    start_time = time.time()
    for curr_epoch in tqdm(range(cfg.n_epochs)):
        # take some steps of the dynamics for online learning
        xgs, prng_key = step_data(xgs, prng_key, cfg)
        xs, gs = onp.split(xgs, 2, axis=1)  # [ntrajs, N, d], [ntrajs, N, d]

        for curr_batch in tqdm(range(cfg.nbatches)):
            ## take a step on the loss
            loss_fn_args, prng_key = setup_loss_fn_args(
                xs, gs, prng_key, curr_batch, cfg
            )
            xbatch, gbatch = loss_fn_args[0], loss_fn_args[1]
            params, opt_state, loss_value, grads = losses.pupdate(
                params, opt_state, opt, loss, loss_fn_args
            )

            ## compute EMA params
            curr_params = unreplicate(params)
            ema_params = update_ema_params(curr_params, ema_params)

            ## log loss, statistics, and score norm
            data = log_metrics(
                data,
                curr_params,
                ema_params,
                xbatch,
                gbatch,
                grads,
                loss_value[0],
                curr_batch,
                curr_epoch,
                cfg,
            )

    # dump one final time
    pickle.dump(data, open(f"{cfg.output_folder}/{cfg.output_name}.npy", "wb"))


#### Initialization ####
def setup_loss(cfg: config_dict.ConfigDict) -> Callable:
    if cfg.loss_type == "score_matching":
        loss = sm_loss
    elif cfg.loss_type == "denoising":
        loss = denoising_loss
    elif cfg.loss_type == "supervised":
        loss = supervised_loss
    else:
        raise ValueError("Specified loss is not implemented.")

    # auto-fill cfg for easy use of pmap
    return functools.partial(loss, cfg=cfg)


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
        description="Run a one-d mips simulation from the command line."
    )
    parser.add_argument("--ntrajs", type=int)
    parser.add_argument("--nsteps_online", type=int)
    parser.add_argument("--load_data", type=int)
    parser.add_argument("--network_path", type=str)
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--bs_stats", type=int)
    parser.add_argument("--bs_online", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dt_online", type=float)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--lamb1", type=float)
    parser.add_argument("--lamb2", type=float)
    parser.add_argument("--lamb3", type=float)
    parser.add_argument("--lamb4", type=float)
    parser.add_argument("--network_type", type=str)
    parser.add_argument("--shift_network", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--embed_n_neurons", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--embed_n_hidden", type=int)
    parser.add_argument("--decode_n_hidden", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--n_layers_feedforward", type=int)
    parser.add_argument("--n_inducing_points", type=int)
    parser.add_argument("--n_neighbors", type=int)
    parser.add_argument("--particle_wise_encode", type=int)
    parser.add_argument("--particle_wise_decode", type=int)
    parser.add_argument("--pool_before_decode", type=int)
    parser.add_argument("--this_particle_pool", type=int)
    parser.add_argument("--network_scale", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--decay_steps", type=int)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--use_skilling_hutch", type=int)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)

    return parser.parse_args()


def setup_config_dict():
    cfg = config_dict.ConfigDict()

    cfg.d = 2
    cfg.r = 1.0
    cfg.gamma = 1e-2  # default parameter -- overrided by arguments.
    cfg.eps = 1e-2  # default parameter -- overrided by arguments.
    cfg.v0 = 0.1  # default parameter -- overrided by arguments.
    cfg.N = 16  # default parameter -- overrided by arguments.
    cfg.A = 0.035  # default parameter -- overrided by arguments.
    cfg.phi = 0.1
    cfg.beta = 7.5
    cfg.thermalize_fac = 20
    cfg.clip = 1.0

    if cfg.gamma == 0:
        cfg.tburn = cfg.thermalize_fac / cfg.eps
    elif cfg.eps == 0:
        cfg.tburn = cfg.thermalize_fac / cfg.gamma
    else:
        cfg.tburn = cfg.thermalize_fac / min(cfg.gamma, cfg.eps)

    cfg.max_n_steps = 50
    cfg.n_epochs = int(1e6)
    cfg.save_freq = int(5e3)
    cfg.visual_freq = int(2.5e3)
    cfg.stat_freq = int(1e3)

    ## input parameters
    args = get_simulation_parameters()
    cfg.nsteps_online = args.nsteps_online
    cfg.loss_type = args.loss_type
    cfg.use_skilling_hutch = args.use_skilling_hutch
    cfg.ntrajs = args.ntrajs
    cfg.v0 = args.v0
    cfg.phi = args.phi
    cfg.A = args.A
    cfg.k = args.k
    cfg.beta = args.beta
    cfg.dt = args.dt
    cfg.dt_online = args.dt_online
    cfg.N = args.N
    cfg.gamma = args.gamma
    cfg.eps = args.eps
    cfg.lamb1 = args.lamb1
    cfg.lamb2 = args.lamb2
    cfg.lamb3 = args.lamb3
    cfg.lamb4 = args.lamb4
    cfg.ema_facs = [0.999, 0.9999]
    cfg.w0 = args.w0
    cfg.shift_network = args.shift_network
    cfg.network_type = args.network_type
    cfg.embed_dim = args.embed_dim
    cfg.embed_n_neurons = args.embed_n_neurons
    cfg.dim_feedforward = cfg.embed_dim
    cfg.num_layers = args.num_layers
    cfg.embed_n_hidden = args.embed_n_hidden
    cfg.decode_n_hidden = args.decode_n_hidden
    cfg.num_heads = args.num_heads
    cfg.n_layers_feedforward = args.n_layers_feedforward
    cfg.n_inducing_points = args.n_inducing_points
    cfg.n_neighbors = args.n_neighbors
    cfg.particle_wise_encode = args.particle_wise_encode
    cfg.particle_wise_decode = args.particle_wise_decode
    cfg.pool_before_decode = args.pool_before_decode
    cfg.this_particle_pool = args.this_particle_pool
    cfg.network_scale = args.network_scale
    cfg.learning_rate = args.learning_rate
    cfg.decay_steps = args.decay_steps
    cfg.wandb_name = f"{args.wandb_name}_{args.slurm_id}"
    cfg.output_folder = args.output_folder
    cfg.output_name = f"{args.output_name}_{args.slurm_id}"
    cfg.bs = args.bs
    cfg.bs_stats = args.bs_stats
    cfg.bs_online = args.bs_online
    cfg.nbatches = int(cfg.ntrajs / cfg.bs)
    cfg.nbatches_stats = int(cfg.bs / cfg.bs_stats)
    cfg.nbatches_online = int(cfg.nbatches / cfg.bs_online)
    cfg.nbatches += 1 if cfg.nbatches * cfg.bs < cfg.ntrajs else 0
    cfg.nbatches_online += 1 if cfg.nbatches_online * cfg.bs_online < cfg.ntrajs else 0
    cfg.nsteps = int(cfg.tburn / cfg.dt + 1)
    cfg.ndevices = jax.local_device_count()

    ## based on input parameters
    cfg.dim = 2 * cfg.N * cfg.d
    cfg.width = float(np.sqrt(cfg.N * np.pi / cfg.phi) / 2)
    cfg.sig0x, cfg.sig0g = cfg.width / 2, 1.0

    return cfg, args


def construct_network(
    cfg: config_dict.ConfigDict,
) -> Tuple[Callable, Callable, Callable, Callable]:
    shift_func, particle_div_shift_func, div_shift_func = define_shift_functions(cfg)

    if "transformer" in cfg.network_type:
        (
            score_net,
            particle_div_net,
            div_net,
        ) = networks.define_full_particle_split_transformer(
            cfg.w0,
            cfg.d,
            cfg.N,
            cfg.num_layers,
            cfg.embed_dim,
            cfg.embed_n_hidden,
            cfg.decode_n_hidden,
            cfg.embed_n_neurons,
            cfg.num_heads,
            cfg.dim_feedforward,
            cfg.n_layers_feedforward,
            cfg.n_inducing_points,
            shift_func,
            particle_div_shift_func,
            div_shift_func,
        )

    elif cfg.network_type == "mlp":
        score_net, particle_div_net, div_net = networks.define_full_particle_split_mlp(
            cfg.w0, cfg.d, cfg.N, cfg.embed_n_hidden, cfg.embed_n_neurons
        )

    else:
        raise ValueError("Network type not defined!")

    map_score_net = jax.jit(
        jax.vmap(score_net.apply, in_axes=(None, 0, 0, None)), static_argnums=3
    )

    return score_net, particle_div_net, div_net, map_score_net


def initialize_network(prng_key: np.ndarray):
    if args.network_path != "":
        loaded_dict = pickle.load(open(args.network_path, "rb"))

        try:
            print("For backwards compatibility, loading from params_list.")
            params = deepcopy(
                jax.device_put(loaded_dict["params_list"][-1], jax.devices("cpu")[0])
            )
        except:
            print("Loading params directly.")
            params = deepcopy(
                jax.device_put(loaded_dict["params"], jax.devices("cpu")[0])
            )

        # ensure we clear the memory
        del loaded_dict
    else:
        ex_xs, ex_gs = np.split(xgs[0], 2)
        key1, key2 = jax.random.split(prng_key)
        prng_key = jax.random.split(key1)[0]
        params = {
            "x": score_net.init(key1, ex_xs, ex_gs, "x"),
            "g": score_net.init(key2, ex_xs, ex_gs, "g"),
        }

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


def define_shift_functions(
    cfg: config_dict.ConfigDict,
) -> Tuple[Callable, Callable]:
    if cfg.shift_network:
        if cfg.n_neighbors > 0:

            def shift_func(
                xs: np.ndarray, gs: np.ndarray, ii: int, key: str
            ) -> np.ndarray:
                if key == "x":
                    xi = xs[ii]
                    return (
                        cfg.k
                        * drifts.single_particle_elastic_interaction(
                            xi, xs, cfg.r, cfg.N, cfg.width, cfg.beta
                        )
                        - cfg.A * xi
                    ) / cfg.eps
                elif key == "g":
                    return -gs[ii]
                else:
                    raise ValueError("Key must be x or g.")

            def particle_div_shift_func(
                xs: np.ndarray, gs: np.ndarray, ii: int, key: str
            ) -> np.ndarray:
                if key == "x":
                    return div_particle_force(xs, ii, cfg) / cfg.eps
                elif key == "g":
                    return -cfg.d
                else:
                    raise ValueError("Key must be x or g.")

            div_shift_func = None
        else:

            def shift_func(xs: np.ndarray, gs: np.ndarray, key: str) -> np.ndarray:
                # v=0 at initialization
                if key == "x":
                    xgs = np.concatenate((xs, gs))
                    return calc_xdots(xgs, cfg) / cfg.eps
                elif key == "g":
                    return -gs
                else:
                    raise ValueError("Key must be x or g.")

            def particle_div_shift_func(
                xs: np.ndarray, gs: np.ndarray, key: str
            ) -> np.ndarray:
                if key == "x":
                    return div_force(xs, cfg) / cfg.eps
                elif key == "g":
                    return -cfg.d * np.ones(cfg.N)
                else:
                    raise ValueError("Key must be x or g.")

            div_shift_func = lambda xs, gs, key: np.sum(
                particle_div_shift_func(xs, gs, key)
            )
    else:
        if cfg.n_neighbors > 0:
            shift_func = particle_div_shift_func = div_shift_func = (
                lambda xs, gs, ii, key: 0.0
            )
        else:
            shift_func = particle_div_shift_func = div_shift_func = (
                lambda xs, gs, key: 0.0
            )

    return shift_func, particle_div_shift_func, div_shift_func


#######################

if __name__ == "__main__":
    ## set up the simulation environment
    cfg, args = setup_config_dict()
    prng_key = jax.random.PRNGKey(onp.random.randint(1000))

    ## generate or load a dataset for learning
    xgs, prng_key, cfg = generate_data(cfg, prng_key, args.load_data)

    ## define and initialize the neural network
    score_net, particle_div_net, div_net, map_score_net = construct_network(cfg)
    params, prng_key = initialize_network(prng_key)
    compute_output_info = functools.partial(
        compute_output_info, score_net=score_net, particle_div_net=particle_div_net
    )

    ## define optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=int(1e4),
        decay_steps=int(cfg.decay_steps),
    )

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip), optax.radam(learning_rate=schedule)
    )

    # for parallel training
    opt_state = replicate(opt.init(params))

    ## set up weights and biases tracking
    wandb.init(
        project="",
        name=cfg.wandb_name,
        config=cfg.to_dict(),
    )

    ## train the model
    data = {
        "params": jax.device_put(params, jax.devices("cpu")[0]),
        "xgs": xgs,
        "cfg": cfg,
    }

    train_loop(prng_key, opt, opt_state, data, cfg)
