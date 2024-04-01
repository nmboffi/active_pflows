"""
Base class for entropy calculation in MIPS.

Nicholas M. Boffi
12/6/22
"""

from dataclasses import dataclass
from typing import Tuple
from copy import deepcopy
import jax
import jax.numpy as np
from jax import vmap, jit
from jax.flatten_util import ravel_pytree
import optax
import numpy as onp
from tqdm.auto import tqdm as tqdm
import dill as pickle
import time
from jaxlib.xla_extension import Device
import haiku as hk
import wandb


# plotting details
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

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
mpl.rcParams["font.size"] = 5.0
mpl.rcParams["legend.fontsize"] = 5.0
mpl.rcParams["figure.dpi"] = 300


State = onp.ndarray
Time = float


from . import drifts
from . import networks
from . import losses
from . import entropy_utils


@dataclass
class MIPSSimStatic:
    """Base class for MIPS calculation, assuming fixed dataset."""

    # optimization parameters
    dataset: dict
    bs: int
    n_epochs: int
    learning_rate: float
    decay_steps: int
    noise_fac: float
    key: np.ndarray
    loss_type: str

    # network parameters
    network_type: str
    scale_fac: float
    w0: float
    ema_fac: float

    # transformer parameters (if applicable)
    n_neighbors: int
    num_layers: int
    embed_dim: int
    embed_n_hidden: int
    embed_n_neurons: int
    num_heads: int
    this_particle_pooling: bool
    dim_feedforward: int

    # logging parameters
    wandb_name: str

    # output information
    dataset_location: str
    output_folder: str
    output_name: str
    save_fac: int
    plot_fac: int

    def __post_init__(self) -> None:
        """Initialize the simulation."""
        self.unpack_dataset()
        print(f"Unpacked the dataset.")
        self.init_network_and_optimizer()
        print(f"Initialized network and optimizer.")
        self.init_loss()
        print(f"Initialized the loss.")
        self.init_wandb()
        print(f"Initialized weights and biases logging.")

    def unpack_dataset(self) -> None:
        # full dataset
        self.data = self.dataset["trajs"]["SDE"]
        print(self.data.shape)

        self.width = self.dataset["width"]
        self.gamma = self.dataset["gamma"]
        self.r = self.dataset["r"]
        self.N = self.dataset["N"]
        self.radii = self.r * np.ones(self.N)
        self.d = self.dataset["d"]
        self.dt = self.dataset["dt"]
        self.v0 = self.dataset["v0"]

        try:
            self.beta = self.dataset["beta"]
        except:
            print("Couldn't find beta! Did you intend for that?")
            self.beta = None
            print("Re-scaling v0 to match old convention...")
            self.v0 /= np.sqrt(self.d)

        self.step_system = lambda xgs, noise: drifts.step_mips_OU_EM(
            xgs,
            self.dt,
            np.ones(self.N),
            0.0,  # A
            1.0,  # k
            self.v0,
            self.N,
            self.d,
            0.0,  # epsilon
            self.gamma,
            self.width,
            self.beta,
            noise,
        )

        # set up batching
        self.n, self.dim = self.data.shape
        self.particle_batch = self.bs <= self.N

        if self.particle_batch:
            self.n_batches = int(self.N / self.bs)
            self.bs_output = self.bs
            self.n_batches_output = self.n_batches
            assert self.N == self.n_batches * self.bs
        else:
            self.bs = self.bs // self.N
            self.n_batches = int(self.n / self.bs)
            self.bs_output = self.N // 2
            self.n_batches_output = 2

    def init_network_and_optimizer(self) -> None:
        """Construct and initialize the neural network and the optimizer."""

        # construct the network
        (
            self.particle_score_net,
            self.particle_div_net,
        ) = networks.define_transformer_networks(
            w0=self.w0,
            n_neighbors=self.n_neighbors,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            embed_n_hidden=self.embed_n_hidden,
            embed_n_neurons=self.embed_n_neurons,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            n_layers_feedforward=1,
            width=self.width,
            network_type=self.network_type,
            this_particle_pooling=self.this_particle_pooling,
            scale_fac=self.scale_fac,
        )

        # construct the entropy comutations.
        _, _, calc_particle_entropies = entropy_utils.define_entropy(
            self.d,
            self.gamma,
            self.width,
            self.beta,
            self.r,
            self.particle_div_net.apply,
        )
        self.calc_particle_entropies = jit(calc_particle_entropies, static_argnums=3)

        # initialize the network
        self.params = self.particle_score_net.init(
            self.key, np.zeros((2 * self.N, self.d)), 0
        )
        _ = self.particle_div_net.init(self.key, np.zeros((2 * self.N, self.d)), 0)
        network_size = ravel_pytree(self.params)[0].size
        print(f"Number of parameters: {network_size}")
        print(f"Loss type: {self.loss_type}. Network type: {self.network_type}.")

        # set up the optimizer
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=int(1e4),
            decay_steps=int(self.decay_steps),
        )

        self.opt = optax.chain(
            optax.clip_by_global_norm(1.0), optax.radam(learning_rate=schedule)
        )

        self.opt_state = self.opt.init(self.params)

        # jit the network and force computations
        self.map_score = jit(
            vmap(self.particle_score_net.apply, in_axes=(None, None, 0))
        )

        self.map_div = jit(vmap(self.particle_div_net.apply, in_axes=(None, None, 0)))

        self.calc_xdots = jit(
            lambda xgs: drifts.mips(
                xgs,
                self.v0,
                0.0,
                1.0,
                self.gamma,
                np.ones(self.N),
                self.width,
                self.beta,
                self.N,
            )[: self.N]
        )

    def init_loss(self) -> None:
        """Define the loss function."""
        if self.loss_type == "score_matching":
            if self.particle_batch:
                print(f"Batching over particles! Batch size: {self.bs}")

                def particle_loss(
                    params: hk.Params, xgs: np.ndarray, ii: int  # [2*N, d]
                ) -> float:
                    score_vals = self.particle_score_net.apply(params, xgs, ii)
                    div_score_vals = self.particle_div_net.apply(params, xgs, ii)
                    return np.sum(score_vals**2) + 2 * div_score_vals

                in_axes = (None, None, 0)
                self.loss = lambda params, xgs, batch_inds: np.mean(
                    vmap(particle_loss, in_axes)(params, xgs, batch_inds)
                )
            else:
                print(f"Batching over snapshots! Batch size: {self.bs}")

                def snapshot_loss(
                    params: hk.Params, xgs: np.ndarray  # [2*N, d]
                ) -> float:
                    score_vals = vmap(
                        self.particle_score_net.apply, in_axes=(None, None, 0)
                    )(params, xgs, np.arange(self.N))

                    div_score_vals = vmap(
                        self.particle_div_net.apply, in_axes=(None, None, 0)
                    )(params, xgs, np.arange(self.N))

                    return np.mean(np.sum(score_vals**2, axis=1) + 2 * div_score_vals)

                self.loss = lambda params, xgs_batch: np.mean(
                    vmap(snapshot_loss, in_axes=(None, 0))(params, xgs_batch)
                )

        # denoising...
        elif "denoising" in self.loss_type:
            if self.loss_type == "denoising_g":

                def construct_pxgs(
                    xgs: np.ndarray, zetas: np.ndarray, ii: int
                ) -> np.ndarray:
                    xs, gs = np.split(xgs, 2)
                    pgs = gs + self.noise_fac * zetas
                    return np.concatenate((xs, pgs)), zetas[ii]

            elif self.loss_type == "denoising_xg":

                def construct_pxgs(
                    xgs: np.ndarray, zetas: np.ndarray, ii: int
                ) -> np.ndarray:
                    return xgs + self.noise_fac * zetas, zetas[self.N + ii]

            def particle_loss(
                params: hk.Params,
                xgs: np.ndarray,  # [2*N, d]
                zetas: np.ndarray,  # [2*N, d] or [N, d], depending on loss_type
                ii: int,
            ) -> float:
                """Particle-wise denoising loss to reduce memory requirements.
                Only apply noise in the g variables.

                Args:
                    params: neural network parameters.
                    xgs:    system state.
                    zetas:  noise.
                    ii:     particle index.
                """
                loss = 0
                for sign in [-1, 1]:
                    pxgs, zeta_gi = construct_pxgs(xgs, sign * zetas, ii)
                    score = self.particle_score_net.apply(
                        params, pxgs, ii
                    )  # d-dimensional
                    loss += np.sum(score**2 + 2 * score * zeta_gi / self.noise_fac)

                return loss

            self.particle_loss = particle_loss
            in_axes = (None, None, None, 0)
            self.loss = lambda params, xgs, zetas, batch_inds: np.mean(
                vmap(self.particle_loss, in_axes)(params, xgs, zetas, batch_inds)
            )

        else:
            raise ValueError(f"Unrecognized loss type: {self.loss_type}")

    def init_wandb(self) -> None:
        self.config = {
            "dataset_location": self.dataset_location,
            "width": self.width,
            "gamma": self.gamma,
            "beta": self.beta,
            "v0": self.v0,
            "r": self.r,
            "N": self.N,
            "d": self.d,
            "n": self.n,
            "dim": self.dim,
            "n_epochs": self.n_epochs,
            "n_batches": self.n_batches,
            "bs": self.bs,
            "learning_rate": self.learning_rate,
            "decay_steps": self.decay_steps,
            "noise_fac": self.noise_fac,
            "w0": self.w0,
            "n_neighbors": self.n_neighbors,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "embed_n_hidden": self.embed_n_hidden,
            "embed_n_neurons": self.embed_n_neurons,
            "num_heads": self.num_heads,
            "this_particle_pooling": self.this_particle_pooling,
            "dim_feedforward": self.dim_feedforward,
            "scale_fac": self.scale_fac,
            "save_fac": self.save_fac,
            "network_type": self.network_type,
            "loss_type": self.loss_type,
            "output_folder": self.output_folder,
            "output_name": self.output_name,
        }

        print(f"Initializing wandb with config: {self.config}")

        wandb.init(
            project="",
            name=self.wandb_name,
            config=self.config,
        )

    def compute_output_info(self, index: int, params: hk.Params):
        """Compute the entropy, activity, etc."""
        xgs = self.data[index].reshape((2 * self.N, self.d))
        xs, gs = np.split(xgs, 2)
        particle_scores = onp.zeros((self.N, self.d))

        total_entropies, x_entropies, g_entropies = (
            onp.zeros(self.N),
            onp.zeros(self.N),
            onp.zeros(self.N),
        )

        for curr_batch in range(self.n_batches_output):
            lb = curr_batch * self.bs_output
            ub = lb + self.bs_output
            batch_inds = np.arange(lb, ub)

            # compute the score
            particle_scores[batch_inds] = self.map_score(params, xgs, batch_inds)

            # compute the entropies
            x_entropies[batch_inds] = self.calc_particle_entropies(
                params, xgs, batch_inds, True
            )
            g_entropies[batch_inds] = self.calc_particle_entropies(
                params, xgs, batch_inds, False
            )

        # construct other visual quantities
        total_entropies = x_entropies + g_entropies
        xdots = self.calc_xdots(xgs)

        gdots = -self.gamma * (gs + particle_scores)
        gdot_mags = np.linalg.norm(gdots, axis=1)
        xdot_mags = np.linalg.norm(xdots, axis=1)
        particle_score_mags = np.linalg.norm(particle_scores, axis=1)

        return (
            xs,
            gs,
            gdot_mags,
            xdot_mags,
            particle_score_mags,
            total_entropies,
            x_entropies,
            g_entropies,
        )

    def make_plot(self, params: hk.Params, index: int):
        # compute data for plotting
        (
            xs,
            gs,
            gdot_mags,
            xdot_mags,
            particle_score_mags,
            total_entropies,
            x_entropies,
            g_entropies,
        ) = self.compute_output_info(index, params)

        # shift to center of mass frame
        center_of_mass = np.mean(xs, axis=0)
        centered_xs = jax.vmap(
            lambda x: drifts.torus_project(x - center_of_mass, self.width)
        )(xs)

        # individual panels
        titles = [
            [r"$\Vert\dot{g}\Vert$", r"$\Vert\dot{x}\Vert$", r"$\Vert s \Vert$"],
            [r"$\nabla_g \cdot v_g$", r"$\nabla_x \cdot v_x$", r"$\nabla\cdot v$"],
        ]

        cs = [
            [gdot_mags, xdot_mags, particle_score_mags],
            [g_entropies, x_entropies, total_entropies],
        ]

        cmaps = [
            sns.color_palette("mako", as_cmap=True),
            sns.color_palette("mako", as_cmap=True),
        ]

        # plot parameters
        plt.close("all")
        plt.style.use("dark_background")
        sns.set_palette("deep")
        nrows = len(titles)
        ncols = len(titles[0])
        fw, fh = 5, 5
        scale_fac = 0.25
        fraction = 0.15
        shrink = 0.75
        fontsize = 12.5

        # create figure
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
            ax.set_xlim([-self.width, self.width])
            ax.set_ylim([-self.width, self.width])
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

                vmin = onp.quantile(c, q=0.05)
                vmax = onp.quantile(c, q=0.95)

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                scat = ax.scatter(
                    centered_xs[:, 0],
                    centered_xs[:, 1],
                    s=(scale * scale_fac * self.r) ** 2,
                    marker="o",
                    c=c,
                    cmap=cmaps[ii],
                    norm=norm,
                )
                cbar = fig.colorbar(scat, ax=ax, fraction=fraction, shrink=shrink)
                cbar.ax.tick_params(labelsize=fontsize)

        wandb.log({"entropy_figure": wandb.Image(fig)})

    def particle_batch_body(
        self,
        curr_epoch: int,
        ts: np.ndarray,
        params: hk.Params,
        opt_state: optax.OptState,
        cpu: Device,
        gpu: Device,
    ) -> Tuple:
        N_inds = onp.arange(self.N)
        n_noises = self.N if self.loss_type == "denoising_g" else 2 * self.N
        ema_params = deepcopy(params)

        with tqdm(range(self.n)) as subbar:
            for curr_iter, tind in enumerate(subbar):
                tt = ts[tind]
                xgs = jax.device_put(self.data[tt].reshape((2 * self.N, self.d)), gpu)
                onp.random.shuffle(N_inds)

                start_time = time.time()
                for curr_batch in range(self.n_batches):
                    lb = curr_batch * self.bs
                    ub = lb + self.bs
                    batch_inds = N_inds[lb:ub]

                    if self.loss_type == "score_matching":
                        loss_fn_args = (xgs, batch_inds)
                    else:
                        zetas = jax.device_put(onp.random.randn(n_noises, self.d), gpu)
                        loss_fn_args = (xgs, zetas, batch_inds)

                    params, opt_state, loss_value, grads = losses.update(
                        params, opt_state, self.opt, self.loss, loss_fn_args
                    )

                    ema_params = update_ema_params(params, ema_params, self.ema_fac)

                    wandb.log(
                        {
                            "loss": loss_value,
                            "grad_norm": losses.compute_grad_norm(grads),
                        }
                    )
                    scores = self.map_score(params, xgs, batch_inds)
                    score_norm = np.mean(np.sum(scores**2, axis=1))
                    wandb.log({"score_norm:": -score_norm})

                if curr_iter % self.save_fac == 0:
                    self.save_data(curr_epoch * self.n + curr_iter, params, ema_params)

                if curr_iter % self.plot_fac == 0:
                    self.make_plot(params, tt)

        return params, ema_params, opt_state

    def snapshot_batch_body(
        self,
        curr_epoch: int,
        ts: np.ndarray,
        params: hk.Params,
        opt_state: optax.OptState,
        cpu: Device,
        gpu: Device,
    ) -> Tuple:
        ema_params = deepcopy(params)

        with tqdm(range(self.n_batches)) as subbar:
            for curr_batch in subbar:
                start_time = time.time()
                lb = curr_batch * self.bs
                ub = lb + self.bs
                batch_ts = ts[lb:ub]
                batch_xgs = jax.device_put(
                    self.data[batch_ts].reshape((batch_ts.size, 2 * self.N, self.d)),
                    gpu,
                )

                loss_fn_args = (batch_xgs,)

                params, opt_state, loss_value, grads = losses.update(
                    params, opt_state, self.opt, self.loss, loss_fn_args
                )

                ema_params = update_ema_params(params, ema_params, self.ema_fac)

                wandb.log(
                    {"loss": loss_value, "grad_norm": losses.compute_grad_norm(grads)}
                )

                if self.loss_type == "score_matching":
                    scores = self.map_score(params, batch_xgs[0], np.arange(self.N))
                    score_norm = np.mean(np.sum(scores**2, axis=1))
                    wandb.log({"score_norm:": -score_norm})

                if curr_batch % self.save_fac == 0:
                    self.save_data(
                        curr_epoch * self.n_batches + curr_batch, params, ema_params
                    )

                if curr_batch % self.plot_fac == 0:
                    self.make_plot(params, batch_ts[0])

        return params, opt_state

    def learn_mips_score(self, cpu: Device, gpu: Device) -> None:
        """Perform the minimization procedure to learn the score on the MIPS data."""
        params = jax.device_put(self.params, gpu)
        opt_state = jax.device_put(self.opt_state, gpu)
        ts = onp.arange(self.n)

        with tqdm(range(self.n_epochs)) as pbar:
            for curr_epoch in pbar:
                pbar.set_description(f"Epoch {curr_epoch}.")
                if self.particle_batch:
                    params, ema_params, opt_state = self.particle_batch_body(
                        curr_epoch, ts, params, opt_state, cpu, gpu
                    )
                else:
                    params, opt_state = self.snapshot_batch_body(
                        curr_epoch, ts, params, opt_state, cpu, gpu
                    )

        self.save_data(-1, params, ema_params)

    def save_data(
        self, save_index: int, params: hk.Params, ema_params: hk.Params
    ) -> None:
        output_data = {
            "config": self.config,
            "params": params,
            "ema_params": ema_params,
        }

        pickle.dump(
            output_data,
            open(f"{self.output_folder}/{self.output_name}_{save_index}.npy", "wb"),
        )


@jax.jit
def update_ema_params(
    params: hk.Params,
    ema_params: hk.Params,
    ema_fac: float,
) -> hk.Params:
    return jax.tree_util.tree_map(
        lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
        params,
        ema_params,
    )
