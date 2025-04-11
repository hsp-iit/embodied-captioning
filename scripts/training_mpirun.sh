#!/bin/bash
#PBS -l select=1:ngpus=1:ncpus=5:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /work/tapicella/code/SImCa/finetuning_blip2_triplet1.txt
#PBS -N simca
#PBS -q gpu_a100

# Navigate to code
cd /work/tapicella/code/SImCa/third_parties/hf-transformers/src

# Activate conda environment
module load miniforge/24.3.0
source activate base
conda activate simca

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

python -m transformers.finetune_models_wandb ++dataset.train_csv="/projects/simca/extracted_dataset/experiments/training/gibson_randomGoal_blip2_mask2former_train_filtered_detection_llm_unique.csv" ++dataset.val_csv="/projects/simca/extracted_dataset/experiments/training/gibson_randomGoal_blip2_mask2former_val_filtered_detection_llm_unique.csv" ++dataset.use_augmentation=True ++training_setup.use_triplet=True ++training_setup.triplet_loss_weight=1.0
