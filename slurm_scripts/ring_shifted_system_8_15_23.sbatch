#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/nb3397/results/mips/2d_ring/8_15_23/output/%A_%a.out
#SBATCH --error=/scratch/nb3397/results/mips/2d_ring/8_15_23/output/%A_%a.err


## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/nb3397/technical/singularity_setup/jax.ext3:ro
PY_DIR=/scratch/nb3397/projects/mips_project/mips/py/launchers
output=/scratch/nb3397/results/mips/2d_ring/8_15_23
launcher_file=twod_ring_simulation.py


# remove all modules
module purge
loss_type="score_matching"


ind=$SLURM_ARRAY_TASK_ID
lamb4s=(1.0 5.0 10.0)
shift_networks=(0 1)


n_lamb4s=${#lamb4s[@]}
n_shift_networks=${#shift_networks[@]}


i=$((ind % n_lamb4s))
j=$((ind / n_lamb4s))

lamb4=${lamb4s[$i]}
shift_network=${shift_networks[$j]}


singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	/bin/bash -c "source /ext3/env.sh; \
	cd $PY_DIR; \
	python $launcher_file \
	--ntrajs 1000000 \
	--nsteps_online 250 \
	--network_path '' \
	--bs 4096 \
	--bs_online 10000 \
	--plot_bs 100000 \
	--dt 0.01 \
	--dt_online 0.01 \
	--w0 0 \
	--v0 0.25 \
	--A 0.0 \
	--k 2.5 \
	--beta 7.5 \
	--width 8.0 \
	--gamma 0.1 \
	--eps 0.1 \
	--lamb1 1.0 \
	--lamb2 0.0 \
	--lamb3 0.0 \
	--lamb4 $lamb4 \
	--shift_network $shift_network \
	--shift_system 1 \
	--symmetric_network 1 \
	--clip_force 0 \
	--smooth_force 1 \
	--n_hidden 4 \
	--n_neurons 256 \
	--learning_rate 0.000001 \
	--decay_steps 200000 \
	--wandb_name 'ring_8_15_23_shifted_system_symmetric' \
	--output_name 'ring_8_15_23_shifted_system_symmetric' \
	--output_folder $output \
	--slurm_id $ind"
exit
