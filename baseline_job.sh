#!/bin/bash
#PBS -N LF_SAM2_baseline
#PBS -P io92
#PBS -q gpuvolta
#PBS -l walltime=10:30:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=50GB
#PBS -l storage=gdata/dk92+scratch/io92+gdata/io92

export TORCH_CUDA_ARCH_LIST="7.0 8.0"
module load cuda/12.5.1
module load python3/3.10.4

cd /g/data/io92/nikolai/LF_SAM_segmentation
source venv/bin/activate
python experiments.py > $PBS_JOBID.log 2>&1
deactivate

