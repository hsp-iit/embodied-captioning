#!/bin/bash
#PBS -l select=1:ngpus=1:ncpus=4:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /work/tapicella/code/SImCa/blip2_vanilla_scores.txt
#PBS -N simca
#PBS -q gpu

# Navigate to code
cd /work/tapicella/code/SImCa

# Activate conda environment
module load miniforge/24.3.0
source activate base
conda activate simca

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

python -m scripts.compute_performance_measures --src_path "/projects/simca/extracted_dataset/experiments/testing/blip2_vanilla_test_set_results_temp_filtered.csv" --dst_path "/projects/simca/extracted_dataset/experiments/testing/blip2_vanilla_test_set_results_filtered.csv"
