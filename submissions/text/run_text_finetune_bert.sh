#!/bin/bash
#$ -cwd                 
#$ -pe smp 4
#$ -l h_rt=1:0:0
#$ -l h_vmem=24G
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_New/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_New/loge/

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

python -m src.pipelines.text_based.finetune_bert --config_filepath src/pipelines/text_based/configs/fine-tune-bert2.yaml

echo "Pipeline finished."
deactivate



