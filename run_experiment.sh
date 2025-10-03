#!/bin/bash
#$ -cwd                 
#$ -pe smp 4
#$ -l h_rt=24:0:0
#$ -l h_vmem=1G
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

python -m src.experiments.run \
    --config_name cprd_lstm_test \
    --experiment_name exp_001 \
echo "Pipeline finished."
deactivate
