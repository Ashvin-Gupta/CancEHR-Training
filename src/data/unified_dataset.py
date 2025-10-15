import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
from tqdm import tqdm

class UnifiedEHRDataset(Dataset):
    """
    A unified, flexible dataset for EHR-based cancer classification.

    This 'smart' dataset can:
    1.  Provide data as integer tokens for custom models ('tokens' format).
    2.  Provide data as natural language text for LLM fine-tuning ('text' format).
    3.  Dynamically truncate patient timelines based on a specified cutoff window
        before the cancer diagnosis date.
    """
    def __init__(self, data_dir, vocab_file, labels_file, medical_lookup_file, lab_lookup_file,
                 cutoff_months=None, max_sequence_length=512, format='tokens', split='train'):
        
        self.format = format
        self.cutoff_months = cutoff_months
        self.max_sequence_length = max_sequence_length
        
        # Load all necessary mappings and lookup tables
        self._load_mappings(vocab_file, labels_file, medical_lookup_file, lab_lookup_file)
        # Load the patient records from the .pkl files for the specified split
        self.patient_records = self._load_data(os.path.join(data_dir, split))

    def _load_mappings(self, vocab_file, labels_file, medical_lookup_file, lab_lookup_file):
        """Loads all vocabularies, translation lookups, and label information."""
        print("Loading all necessary mappings...")
        
        vocab_df = pd.read_csv(vocab_file)
        self.id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        if self.format == 'text':
            medical_df = pd.read_csv(medical_lookup_file)
            self.medical_lookup = pd.Series(medical_df['term'].values, index=medical_df['code'].astype(str).str.upper()).to_dict()
            
            lab_df = pd.read_csv(lab_lookup_file)
            self.lab_lookup = pd.Series(lab_df['term'].values, index=lab_df['code'].astype(str).str.upper()).to_dict()

        labels_df = pd.read_csv(labels_file)
        labels_df['string_label'] = labels_df.apply(lambda row: 'Control' if row['is_case'] == 0 else row['site'], axis=1)
        
        unique_labels = sorted([l for l in labels_df['string_label'].unique() if l != 'Control'])
        self.label_to_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        self.label_to_id_map['Control'] = 0
        
        print("Created label-to-ID mapping:")
        print(self.label_to_id_map)
        
        labels_df['label_id'] = labels_df['string_label'].map(self.label_to_id_map)
        self.subject_to_label = pd.Series(labels_df['label_id'].values, index=labels_df['subject_id']).to_dict()
        
        labels_df['cancerdate'] = pd.to_datetime(labels_df['cancerdate'], errors='coerce')
        self.subject_to_cancer_date = pd.Series(labels_df['cancerdate'].values, index=labels_df['subject_id']).to_dict()


    def _load_data(self, data_dir):
        """Loads all patient records from .pkl files in a directory."""
        records = []
        pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
        for file_path in tqdm(pkl_files, desc=f"Loading data from {data_dir}"):
            with open(file_path, 'rb') as f:
                records.extend(pickle.load(f))
        return records

    def _translate_token(self, token_string):
        # This logic is the same as our narrative generator
        if not isinstance(token_string, str): return ""
        try:
            if token_string.startswith('<time_interval_'):
                time_part = token_string.split('_')[-1].strip('>')
                return f"{time_part}"
            elif token_string.startswith('AGE_'):
                return f"{token_string}"
            elif token_string.startswith('MEDICAL//'):
                code = token_string.split('//')[1].upper()
                return self.medical_lookup.get(code, code.replace('_', ' ').title())
            elif token_string.startswith('MEASUREMENT//'):
                code = token_string.split('//')[1].upper()
                description = self.medical_lookup.get(code, code.replace('_', ' ').title())
                return f"{description}"
            elif token_string.startswith('LAB//'):
                code = token_string.split('//')[1].upper()
                return self.lab_lookup.get(code, code.replace('_', ' ').title())
            elif token_string.startswith(('BMI//', 'HEIGHT//', 'WEIGHT//')):
                return f"{token_string.split('//')[0]}: {token_string.split('//')[1]}"
            elif token_string.startswith(('GENDER//', 'ETHNICITY//')):
                parts = token_string.split('//')
                return f"{parts[1]}"
            elif token_string.startswith('REGION//'):
                parts = token_string.split('//')
                return f"{parts[1]}"
            elif token_string.startswith('Q') and len(token_string) <= 4 and token_string[1:].isdigit():
                return f"{token_string[1:]}"
            elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return ""
            else:
                return f"Unknown"
        except Exception as e:
            print(f"--- DEBUG: FAILED TO TRANSLATE TOKEN ---")
            print(f"Problematic Token String: '{token_string}'")
            print(f"Error: {e}")
            return "---ERROR_TRANSLATING---"

    def __len__(self):
        return len(self.patient_records)

    def __getitem__(self, idx):
        patient_record = self.patient_records[idx]
        subject_id = patient_record['subject_id']
        
        label = self.subject_to_label.get(subject_id)
        if pd.isna(label):
            return None # Skip patients without labels
            
        token_ids = patient_record['tokens']
        timestamps = patient_record['timestamps']
        
        # --- DYNAMIC TIME TRUNCATION ---
        if self.cutoff_months is not None and label > 0: # Only truncate cancer cases
            cancer_date = self.subject_to_cancer_date.get(subject_id)
            if pd.notna(cancer_date):
                cutoff_date = cancer_date - pd.DateOffset(months=self.cutoff_months)
                cutoff_timestamp = cutoff_date.timestamp()
                
                truncated_ids = []
                for i, ts in enumerate(timestamps):
                    if ts == 0 or (ts is not None and ts < cutoff_timestamp):
                        truncated_ids.append(token_ids[i])
                token_ids = truncated_ids
        
        # --- CONTEXT WINDOW TRUNCATION ---
        if self.max_sequence_length is not None and self.format == 'tokens':
            if len(token_ids) > self.max_sequence_length:
                token_ids = token_ids[-self.max_sequence_length:]

        
        # --- FORK THE OUTPUT FORMAT ---
        if self.format == 'tokens':
            return {
                "tokens": torch.tensor(token_ids, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }
        elif self.format == 'text':
            string_codes = [self.id_to_token_map.get(tid, "") for tid in token_ids]
            translated_phrases = [self._translate_token(code) for code in string_codes]
            narrative = ", ".join([phrase for phrase in translated_phrases if phrase])
            return {
                "text": narrative,
                "label": torch.tensor(label, dtype=torch.long)
            }
        else:
            raise ValueError(f"Invalid format specified: {self.format}")