#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/embed_%j.out

module purge
module load 2023
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# Optional: activate your conda or venv
source myenv/bin/activate

python embed_texts.py \
    --input data/answers.csv \
    --output emb \
    --model NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers
