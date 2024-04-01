"""
Nicholas M. Boffi
4/1/24

Generate a dataset for the MIPS system.
"""

import jax
import numpy as onp
from jax import numpy as np
from jax import vmap, jit
import dill as pickle
import sys

sys.path.append("../")
import common.drifts as drifts
import argparse
import time
from functools import partial
from tqdm.auto import tqdm as tqdm


@partial(jit, static_argnums=(6, 7))
def rollout(
    init_xg: np.ndarray,  # [2N, d]
    noises: np.ndarray,  # [nsteps, N, d]
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
) -> np.ndarray:
    """Rollout the MIPS system for nsteps steps."""
    print("Jitting the rollout...")

    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = drifts.step_mips_OU_EM(
            xg, dt, radii, A, k, v0, N, d, eps, gamma, width, beta, noise
        )
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--ndata", type=int)
    parser.add_argument("--nbatches", type=int)
    parser.add_argument("--nbatches_space", type=int)
    parser.add_argument("--tspace", type=float)
    parser.add_argument("--thermalize_fac", type=float)
    parser.add_argument("--load_thermalized", type=int)
    parser.add_argument("--thermalized_location", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    ## standard system parameters
    print("Entering main. Loading command line arguments.")
    d = 2
    r = 1.0
    n_prints = 100

    ## command line arguments
    args = get_cmd_arguments()
    gamma = args.gamma
    N = args.N
    v0 = args.v0
    A = args.A
    k = args.k
    eps = args.eps
    beta = args.beta
    phi = args.phi
    dt = args.dt
    ndata = args.ndata
    output_folder = args.output_folder
    slurm_id = args.slurm_id
    radii = onp.ones(N) * r
    width = np.sqrt(np.sum(radii**2) * np.pi / phi) / 2
    dim = 2 * N * d
    sig0x, sig0g = width / 2, 1.0

    ## thermalization parameters
    nbatches = args.nbatches
    nbatches_space = args.nbatches_space
    thermalize_fac = args.thermalize_fac
    tspace = args.tspace
    divide_fac = min(gamma, eps)
    divide_fac = max(gamma, eps) if divide_fac == 0 else divide_fac
    tf = thermalize_fac / divide_fac
    nsteps_thermalize = int((tf / dt) // nbatches) + 1
    nsteps_space = int((tspace / dt) // nbatches_space) + 1
    key = jax.random.PRNGKey(onp.random.randint(10000))

    ## set up the trajectory storage
    traj = onp.zeros((ndata + 1, 2 * N, d))

    # uniform initial conditions for gas
    if phi < 0.25:
        xs = drifts.torus_project(
            width * onp.random.uniform(-1, 1, size=(N * d)), width
        )
    # gaussian for solids
    else:
        xs = drifts.torus_project(sig0x * onp.random.randn(N * d), width)

    gs = sig0g * onp.random.randn(N * d)
    xgs = onp.concatenate((xs, gs)).reshape((2 * N, d))
    name = f"OU_v0={v0}_N={N}_gamma={gamma}_phi={phi}_dt={dt}_beta={beta}_tspace={tspace}_A={A}_k={k}_eps={eps}"

    # log some info
    print(f"Starting dataset generation.")
    print(f"Output: {output_folder}/{name}.npy")

    # set up output data storage
    data_dict = {
        "gamma": gamma,
        "eps": eps,
        "v0": v0,
        "k": k,
        "A": A,
        "beta": beta,
        "phi": phi,
        "dt": dt,
        "tspace": tspace,
        "tf_thermalize": tf,
        "width": width,
        "r": r,
        "d": d,
        "N": N,
    }

    ## thermalize
    if args.load_thermalized:
        print("Loading thermalized data.")
        thermalized_data = pickle.load(open(args.thermalized_location, "rb"))
        xgs = thermalized_data["traj"][-1]
    else:
        for curr_batch in tqdm(range(nbatches)):
            print(f"Starting thermal batch {curr_batch+1}/{nbatches}")
            batch_start = time.time()
            noises = jax.random.normal(key, shape=(nsteps_thermalize, 2 * N, d))
            key = jax.random.split(key)[0]
            xgs = rollout(
                xgs, noises, dt, radii, A, k, v0, N, d, eps, gamma, width, beta
            )
            batch_end = time.time()
            print(
                f"Finished thermal batch {curr_batch+1}/{nbatches}. Time: {(batch_end-batch_start)/60}m."
            )
        print(f"Finished thermalizing.")

    ## temporal dataset
    start_time = time.time()
    traj[0] = xgs
    for curr_datapt in tqdm(range(ndata)):
        for curr_batch in range(nbatches_space):
            noises = jax.random.normal(key, shape=(nsteps_space, 2 * N, d))
            key = jax.random.split(key)[0]
            xgs = rollout(
                xgs, noises, dt, radii, A, k, v0, N, d, eps, gamma, width, beta
            )
        traj[curr_datapt + 1] = xgs

        try:
            if (curr_datapt % int(ndata // n_prints)) == 0:
                end_time = time.time()
                print(f"Finished data point {curr_datapt+1}/{ndata}.")
                print(f"Total time: {(end_time - start_time)/60}m.")
                start_time = time.time()
                data_dict["traj"] = traj
                pickle.dump(data_dict, open(f"{output_folder}/{name}.npy", "wb"))
        except:
            print("Too few data points to print progress.")

    data_dict["traj"] = traj
    pickle.dump(data_dict, open(f"{output_folder}/{name}_{slurm_id}.npy", "wb"))
