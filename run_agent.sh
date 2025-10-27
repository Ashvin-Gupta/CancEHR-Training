#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0       # 2 hours per run (1 epoch), 1 for debugging
#$ -l h_vmem=11G
#$ -l gpu=1

# 1. Activate your environment
source /data/home/qc25022/miniconda3/bin/activate /data/home/qc25022/miniconda3/envs/llm

# 2. Define your Sweep ID (get this from the 'wandb sweep' command)
SWEEP_ID="ashvingupta00/ehr-llm-pretraining/9eyqcotp"

# 3. Define the *exact* command the agent should run for each job
# This is your Python script + its required --config_path argument
# IMPORTANT: Use the *full path* to your script and config
PYTHON_SCRIPT_PATH="/data/home/qc25022/CancEHR-Training/src/experiments/run_llm_pretrain.py"
CONFIG_FILE_PATH="/data/home/qc25022/CancEHR-Training/src/experiments/configs/llm-pretrain.yaml"

# 4. Run the agent
# It will execute this command over and over with new sweep params
wandb agent --count 10 ${SWEEP_ID} python ${PYTHON_SCRIPT_PATH} --config_path ${CONFIG_FILE_PATH}
