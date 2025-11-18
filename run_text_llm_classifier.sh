#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -l h_vmem=11G
#$ -l gpu=1
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

# Run the fine-tuning script
python -m src.pipelines.text_based.finetune_llm_classifier \
   --config_filepath src/pipelines/text_based/configs/llm_finetune_classifier.yaml
# python -m src.pipelines.text_based.test_classifier_setup --config_filepath src/pipelines/text_based/configs/llm_finetune_classifier.yaml
#     --config_filepath src/pipelines/text_based/configs/llm_finetune_classifier.yaml
echo "Classification fine-tuning complete!"


