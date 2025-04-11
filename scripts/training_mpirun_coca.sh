#!/bin/bash
#PBS -l select=1:ngpus=1:ncpus=5
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /work/tapicella/code/SImCa/finetuning_coca_triplet01.txt
#PBS -N simca
#PBS -q gpu_a100

# Navigate to code
cd /work/tapicella/code/SImCa/third_parties/open_clip/src

# Activate conda environment
module load miniforge/24.3.0
source activate base
conda activate simca

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

python -m open_clip_train.main  --batch-size 64 --coca-triplet-loss-weight 0.1 --train-data /projects/simca/extracted_dataset/experiments/training/gibson_policy_train_coca_llm_filtered_detection_unique_minimum_sample.csv --val-data /projects/simca/extracted_dataset/experiments/training/gibson_policy_val_coca_llm_filtered_detection_unique.csv

