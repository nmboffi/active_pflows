# Learning probability flows and entropy production rates in active matter physics
This repository provides an efficient implementation in ``jax`` of a score matching and physics informed neural network-based algorithm for solving the stationary Fokker-Planck equation in high dimension.

# Installation
The implementation is built on Google's [``jax``](https://github.com/google/jax) package for accelerated linear algebra and DeepMind's [``haiku``](https://github.com/deepmind/dm-haiku) package for neural networks. Both can be installed by following the guidelines at the linked repositories.

# Usage
Routines common to all implemented simulations can be found in ``py/common``, including implementations of the various neural networks used, systems studied, and loss functions used.

Simulation code to launch learning experiments can be found in ``py/launchers``.

Code for generating datasets can be found in ``py/dataset_gen``.

Code for visualizing the output of simulations and for producing the publication figures can be found in ``notebooks``.

Slurm ``sbatch`` scripts used to launch the experiments in the paper can be found under ``slurm_scripts``.

Experiment tracking is implemented in [Weights and Biases](https://wandb.ai/home). You will need to input a project 


# Referencing
If you found this repository useful, please consider citing

[1] N. M. Boffi and Eric Vanden-Eijnden. “Deep learning probability flows and entropy production rates in active matter", arXiv: 2309.12991.


```
@misc{boffi2023deep,
      title={Deep learning probability flows and entropy production rates in active matter}, 
      author={Nicholas M. Boffi and Eric Vanden-Eijnden},
      year={2023},
      eprint={2309.12991},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech}
}
```
