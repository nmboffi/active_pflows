#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/nb3397/results/mips/lowd_cates/8_6_23/output/%A_%a.out
#SBATCH --error=/scratch/nb3397/results/mips/lowd_cates/8_6_23/output/%A_%a.err


## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/nb3397/technical/singularity_setup/jax.ext3:ro
PY_DIR=/scratch/nb3397/projects/mips_project/mips/py/launchers
output=/scratch/nb3397/results/mips/lowd_cates/8_6_23
launcher_file=few_particles_split_score.py


# remove all modules
module purge
loss_type="score_matching"


ind=$SLURM_ARRAY_TASK_ID
v0s=(0.1 0.2 0.3 0.4 0.5)
i=$((ind % 5))
v0=${v0s[$i]}


singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	/bin/bash -c "source /ext3/env.sh; \
	cd $PY_DIR; \
	python $launcher_file \
	--ntrajs 0 \
	--nsteps_online 1000 \
	--load_data 1 \
	--network_path '' \
	--data_folder /scratch/nb3397/projects/mips_project/dataset/lowd_cates/standard_EM \
	--bs 64 \
	--bs_stats 64 \
	--bs_online 10000 \
	--dt 0.001 \
	--dt_online 0.001 \
	--w0 0.0 \
	--N 64 \
	--v0 $v0 \
	--A 0.05 \
	--k 2.5 \
	--beta 7.5 \
	--gamma 0.1 \
	--eps 0.1 \
	--lamb1 1.0 \
	--lamb2 0.0 \
	--lamb3 0.0 \
	--lamb4 1.0 \
	--network_type 'transformer' \
	--shift_network 1 \
	--embed_dim 256 \
	--embed_n_neurons 256 \
	--num_layers 4 \
	--embed_n_hidden 2 \
	--decode_n_hidden 2 \
	--num_heads 4 \
	--n_layers_feedforward 1 \
	--n_inducing_points 0 \
	--n_neighbors 0 \
	--learning_rate 0.0001 \
	--decay_steps 250000 \
	--wandb_name 8_6_23_N64_sm_pinn_only_a100 \
	--output_name 8_6_23_N64_sm_pinn_only_a100 \
	--loss_type $loss_type \
	--output_folder $output \
	--slurm_id $ind"
exit
