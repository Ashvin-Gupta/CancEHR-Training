import argparse
import yaml
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.data.unified_dataset import UnifiedEHRDataset

def create_embedding_corpus(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config['data']
    model_config = config['model']

    # Get base directories
    base_data_dir = data_config['base_data_dir']
    base_output_dir = data_config['embedding_output_dir']
    
    # Define splits to process
    splits = ['train', 'tuning', 'held_out']
    
    # 1. Load E5 model (run this on a GPU node)
    print(f"Loading {model_config['model_name']} embedding model...")
    device = model_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    embed_model = SentenceTransformer(model_config['model_name'], device=device)
    
    # 2. Process each split
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Create output directory for this split
        split_output_dir = os.path.join(base_output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Load dataset for this split
        dataset_args = {
            "data_dir": base_data_dir,
            "vocab_file": data_config["vocab_file"],
            "labels_file": data_config["labels_file"],
            "medical_lookup_file": data_config["medical_lookup_file"],
            "lab_lookup_file": data_config["lab_lookup_file"],
            "format": "events",
            "max_sequence_length": None
        }
        base_dataset = UnifiedEHRDataset(split=split, **dataset_args)

        # Process patients for this split
        print(f"Embedding {len(base_dataset)} patients for {split} split...")
        valid_patients = 0
        
        for i in tqdm(range(len(base_dataset)), desc=f"Embedding {split} split"):
            item = base_dataset[i]
            if item is None:
                continue
                
            # Get the list of event strings
            event_texts = item['events']
            if not event_texts:
                continue # Skip empty records

            # Run the E5 model
            embeddings = embed_model.encode(event_texts, convert_to_tensor=True, device=device)
            
            # Get the label
            label = item['label']
            
            # Save the tensor and label to a file
            output_data = {"embeddings": embeddings.cpu(), "label": label.cpu()}
            torch.save(output_data, os.path.join(split_output_dir, f"patient_{i}.pt"))
            valid_patients += 1

        print(f"Successfully embedded {valid_patients} patients out of {len(base_dataset)} for {split} split")
        print(f"Saved to: {split_output_dir}")

    print(f"\n{'='*60}")
    print("Embedding corpus creation complete!")
    print(f"All splits saved to: {base_output_dir}")
    print(f"Directory structure:")
    for split in splits:
        split_dir = os.path.join(base_output_dir, split)
        if os.path.exists(split_dir):
            file_count = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            print(f"  - {split}/: {file_count} patient files")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    create_embedding_corpus(args.config)