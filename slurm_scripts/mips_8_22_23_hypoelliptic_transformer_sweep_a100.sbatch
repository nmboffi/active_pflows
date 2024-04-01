#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/nb3397/results/mips/hypoelliptic/8_22_23/output/%A_%a.out
#SBATCH --error=/scratch/nb3397/results/mips/hypoelliptic/8_22_23/output/%A_%a.err


## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/nb3397/technical/singularity_setup/jax.ext3:ro
PY_DIR=/scratch/nb3397/projects/mips_project/mips/py/launchers
output=/scratch/nb3397/results/mips/hypoelliptic/8_22_23
launcher_file=run_mips_static.py
data=/scratch/nb3397/projects/mips_project/dataset/big_dataset_5_23_23/combined_dataset.npy


# remove all modules
module purge
ind=$SLURM_ARRAY_TASK_ID


## hyperparameter sweep
learning_rates=(0.0001 0.00001 0.000001)
noise_facs=(0.01 0.001)


## get length of hyperparameter lists
n_learning_rates=${#learning_rates[@]}
n_noise_facs=${#noise_facs[@]}


## get indices from slurm index
i=$((ind % n_learning_rates))
j=$((ind / n_learning_rates % n_noise_facs))


## index arrays to get hyperparameters
learning_rate=${learning_rates[$i]}
noise_fac=${noise_facs[$j]}


## fixed parameters
network_type='transformer_separate_encode'
batch_size=1024
embed_n_hidden=4


singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	/bin/bash -c "source /ext3/env.sh; \
	cd $PY_DIR; \
	python $launcher_file \
	--dataset_location $data \
	--bs $batch_size \
	--n_epochs 1 \
	--loss_type denoising_xg \
	--network_type $network_type \
	--learning_rate $learning_rate \
	--decay_steps 500000 \
	--w0 0.0 \
	--scale_fac 1.0 \
	--noise_fac $noise_fac \
	--output_folder $output \
	--wandb_name hypoelliptic_mips_sweep_trans_ema_8_22_23 \
	--output_name hypoelliptic_mips_sweep_trans_ema_8_22_23 \
	--slurm_id $ind \
	--n_neighbors 64 \
	--num_layers 4 \
	--embed_dim 256 \
	--embed_n_hidden $embed_n_hidden \
	--embed_n_neurons 256 \
	--num_heads 4 \
	--this_particle_pooling 1 \
	--dim_feedforward 256
"
exit
