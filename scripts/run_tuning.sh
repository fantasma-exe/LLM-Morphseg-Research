mkdir -p /userspace/fat/node_logs

#!/bin/bash
#SBATCH --job-name=qwen3-4b-morpheme-seg
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/userspace/fat/node_logs/%x-%j-%N.log
#SBATCH --error=/userspace/fat/node_logs/%x-%j-%N.err

export WORKING_DIR="/userspace/fat"
export PROJECT_DIR="${WORKING_DIR}/LLM-Morphseg-Research"
cd $PROJECT_DIR

export HF_HOME="${WORKING_DIR}/.cache/huggingface"
export UV_CACHE_DIR="${WORKING_DIR}/.cache/uv"
export TMPDIR="${WORKING_DIR}/.tmp"
export PYTHONPYCACHEPREFIX="${WORKING_DIR}/.cache/pycache"

mkdir -p $HF_HOME $UV_CACHE_DIR $TMPDIR $PYTHONPYCACHEPREFIX

export PYTHONUNBUFFERED=1

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

uv run do-train \
    model.model_cfg.model_name=model/Qwen/Qwen3-4b \
    logger=csv \

echo "Job finished at: $(date)"