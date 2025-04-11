#!/bin/bash
#PBS -l select=1:ngpus=1:ncpus=4:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /work/tapicella/code/SImCa/blip2_evaluate_output_error.txt
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

python -m scripts.evaluate_finetuned_model --ckpt_path "/work/tapicella/code/SImCa/third_parties/hf-transformers/src/outputs/2025-02-03/17-49-29/checkpoints/simca/xbvi4os2/checkpoint_7.pt" --model_name "blip2" --test_file "/projects/simca/extracted_dataset/experiments/annotated/gibson_annotated_test_set_filtered.csv" --dest_file "/projects/simca/extracted_dataset/experiments/testing/blip2_vanilla_test_set_results_temp_filtered.csv"
