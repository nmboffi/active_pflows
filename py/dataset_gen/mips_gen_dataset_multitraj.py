import jax
import numpy as onp
from jax import numpy as np
import dill as pickle
import sys

sys.path.append("../")
import common.drifts as drifts
import argparse
import time
from tqdm.auto import tqdm as tqdm
from pathlib import Path


def rollout(
    init_xg: np.ndarray,  # [2N, d]
    noises: np.ndarray,  # [nsteps, N, d]
) -> np.ndarray:
    def scan_fn(xg: np.ndarray, noise: np.ndarray):
        xgnext = drifts.step_mips_OU_EM(
            xg, dt, radii, A, k, v0, N, d, eps, gamma, width, beta, noise
        )
        return xgnext, xgnext

    xg_final, xg_traj = jax.lax.scan(scan_fn, init_xg, noises)
    return xg_final


@jax.jit
def rollout_trajs(
    init_xgs: np.ndarray,  # [ntrajs, 2*N, d]
    noises: np.ndarray,  # [ntrajs, nsteps, 2*N, d]
) -> np.ndarray:
    print("Jitting rollout_trajs!")
    return jax.vmap(lambda init_xg, traj_noises: rollout(init_xg, traj_noises))(
        init_xgs, noises
    )


def generate_data() -> np.ndarray:
    xs = drifts.torus_project(sig0x * onp.random.randn(ntrajs, N, d), width)
    gs = sig0g * onp.random.randn(ntrajs, N, d)
    xgs = onp.concatenate((xs, gs), axis=1)
    key = jax.random.PRNGKey(onp.random.randint(100000))

    start_time = time.time()
    print(f"Starting data generation.")
    for curr_batch in tqdm(range(nbatches)):
        noises = jax.random.normal(key, shape=(ntrajs, nsteps_batch, 2 * N, d))
        xgs = rollout_trajs(xgs, noises)
        key = jax.random.split(key)[0]
    end_time = time.time()
    print(f"Finished data generation. Total time={(end_time-start_time)/60.}m")

    return onp.array(xgs)


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--A", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--ntrajs", type=int)
    parser.add_argument("--nbatches", type=int)
    parser.add_argument("--thermalize_fac", type=float)
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
    ntrajs = args.ntrajs
    radii = onp.ones(N) * r
    width = np.sqrt(np.sum(radii**2) * np.pi / phi) / 2
    dim = 2 * N * d
    name = f"N{N}_v0={v0}_gam={gamma}_eps={eps}_beta={beta}_A={A}_k={k}_dt={dt}"
    output_folder = f"{args.output_folder}/{name}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    sig0x, sig0g = width / 2, 1.0

    ## thermalization parameters
    nbatches = args.nbatches
    thermalize_fac = args.thermalize_fac
    divide_fac = min(gamma, eps)
    divide_fac = max(gamma, eps) if divide_fac == 0 else divide_fac
    tf = thermalize_fac / divide_fac
    nsteps_total = int(tf / dt) + 1
    nsteps_batch = nsteps_total // nbatches

    # generate data and set up storage
    data_dict = {
        "v0": v0,
        "gamma": gamma,
        "eps": eps,
        "phi": phi,
        "dt": dt,
        "beta": beta,
        "A": A,
        "k": k,
        "N": N,
        "tf": tf,
        "width": width,
        "r": r,
        "d": d,
        "xgs": generate_data(),
    }

    print(f"Dumping data to {output_folder}/{args.slurm_id}.npy")
    pickle.dump(data_dict, open(f"{output_folder}/{args.slurm_id}.npy", "wb"))
    print(f"Successfully dumped the data!")
