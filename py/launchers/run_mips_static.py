import jax
import numpy as onp
import sys
import argparse

sys.path.append("../")
import common.mips_sim_static as mips_sim
import dill as pickle


print(jax.devices())
gpu = jax.devices("gpu")[0]
cpu = jax.devices("cpu")[0]


repeatable_seed = False
if repeatable_seed:
    key = jax.random.PRNGKey(42)
    onp.random.seed(42)
else:
    key = jax.random.PRNGKey(onp.random.randint(1000))


# output information
save_fac = 250
plot_fac = 250
ema_fac = 0.9999


def construct_simulation(args):
    dataset = pickle.load(open(args.dataset_location, "rb"))
    output_name = f"{args.output_name}_{args.slurm_id}"

    sim = mips_sim.MIPSSimStatic(
        dataset=dataset,
        bs=args.bs,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        decay_steps=args.decay_steps,
        noise_fac=args.noise_fac,
        key=key,
        loss_type=args.loss_type,
        network_type=args.network_type,
        scale_fac=args.scale_fac,
        w0=args.w0,
        ema_fac=ema_fac,
        n_neighbors=args.n_neighbors,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        embed_n_hidden=args.embed_n_hidden,
        embed_n_neurons=args.embed_n_neurons,
        num_heads=args.num_heads,
        this_particle_pooling=args.this_particle_pooling,
        dim_feedforward=args.dim_feedforward,
        dataset_location=args.dataset_location,
        output_folder=args.output_folder,
        output_name=output_name,
        save_fac=save_fac,
        plot_fac=plot_fac,
        wandb_name=f"{args.wandb_name}_{args.slurm_id}",
    )

    return sim


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
        description="Run a MIPS simulation from the command line."
    )
    parser.add_argument("--dataset_location", type=str, help="Path to dataset.")
    parser.add_argument("--bs", type=int, help="Batch size (in particles).")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--loss_type", type=str, help="What kind of loss?")
    parser.add_argument("--network_type", type=str, help="Network parameterization.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--decay_steps", type=float, help="For cosine annealing.")
    parser.add_argument(
        "--w0", type=float, help="w0 parameter in network architecture."
    )
    parser.add_argument("--scale_fac", type=float, help="Scaling factor.")
    parser.add_argument("--noise_fac", type=float, help="Denoising parameter.")
    parser.add_argument("--output_folder", type=str, help="Path to output.")
    parser.add_argument("--wandb_name", type=str, help="Base name for wandb log.")
    parser.add_argument("--output_name", type=str, help="Base name for output")
    parser.add_argument("--slurm_id", type=str, help="Slurm id.")
    parser.add_argument(
        "--n_neighbors", type=int, help="Number of neighbors for attention."
    )
    parser.add_argument("--num_layers", type=int, help="Layers in transformer.")
    parser.add_argument(
        "--embed_dim", type=int, help="Transformer embedding dimension."
    )
    parser.add_argument(
        "--embed_n_hidden", type=int, help="Transformer layers for embedding."
    )
    parser.add_argument(
        "--embed_n_neurons", type=int, help="Transformer neurons for embedding."
    )
    parser.add_argument("--num_heads", type=int, help="Transformer number of heads.")
    parser.add_argument(
        "--this_particle_pooling", type=int, help="Pool by mean or indexing."
    )
    parser.add_argument(
        "--dim_feedforward", type=int, help="Transformer feedforward dimension."
    )

    return parser.parse_args()


if __name__ == "__main__":
    sim = construct_simulation(get_simulation_parameters())
    sim.learn_mips_score(cpu, gpu)
