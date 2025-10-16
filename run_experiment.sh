#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -l gpu=1
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_Files/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_Files/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/CancEHR-Training"

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

python -m src.experiments.run_hf_finetune \
   --config_filepath src/experiments/configs/fine-tune-bert.yaml 

echo "Pipeline finished."
deactivate
