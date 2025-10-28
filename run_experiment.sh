#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=24:0:0
#$ -l h_vmem=11G
#$ -l gpu=1
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_Files/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_Files/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/CancEHR-Training"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"

# python -m src.experiments.run \
#    --config_name cprd_decoder_lstm_test \
#    --experiment_name exp_001 

python -m src.experiments.run_llm_pretrain --config_filepath src/experiments/configs/llm_pretrain.yaml
# python -m src.experiments.create_embedding_corpus --config_filepath src/experiments/configs/embed_text.yaml

# python -m src.experiments.run_llm --config_filepath src/experiments/configs/llm_pretrain.yaml
echo "Pipeline finished."
deactivate
