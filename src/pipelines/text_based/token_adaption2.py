import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unsloth
from unsloth import FastLanguageModel
import torch.nn.functional as F
import pandas as pd

'''
This script is used to translate the EHR tokens to natural language.
It is in the format where the natural language events are split by special tokens which are then added to the tokenizer.
'''

class EHRTokenTranslator:
    """
    Class to handle translation of EHR tokens to natural language.
    Uses the same translation logic as UnifiedEHRDataset.
    """
    
    def __init__(self, medical_lookup_filepath, lab_lookup_filepath, region_lookup_filepath):
        """
        Initialize the translator with lookup tables.
        
        Args:
            medical_lookup_filepath: Path to medical lookup CSV
            lab_lookup_filepath: Path to lab lookup CSV
        """
        # Load medical and lab lookups
        medical_df = pd.read_csv(medical_lookup_filepath)
        self.medical_lookup = pd.Series(
            medical_df['term'].values, 
            index=medical_df['code'].astype(str).str.upper()
        ).to_dict()
        
        lab_df = pd.read_csv(lab_lookup_filepath)
        self.lab_lookup = pd.Series(
            lab_df['term'].values, 
            index=lab_df['code'].astype(str).str.upper()
        ).to_dict()

        region_df = pd.read_csv(region_lookup_filepath)
        self.region_lookup = pd.Series(
            region_df['Description'].values, 
            index=region_df['regionid'].astype(str).str.upper()
        ).to_dict()
    
    def _translate_token(self, token_string):
        """
        Actually may not need this as Unified Dataset already has the natural language events.
        Translate a single token string to natural language.
        Same logic as UnifiedEHRDataset._translate_token.
        
        Args:
            token_string: The token string to translate
            
        Returns:
            str: Natural language representation, or empty string if skip
        """
        if not isinstance(token_string, str):
            return ""
        
        try:
            if token_string.startswith('<time_interval_'):
                time_part = token_string.split('_')[-1].strip('>')
                return f"<TIME> {time_part}"
            elif token_string.startswith('AGE: '):
                return f"<DEMOGRAPHIC> {token_string}"
            elif token_string.startswith('MEDICAL//BMI'):
                return f"<DEMOGRAPHIC> {token_string.split('//')[1]}"
            elif token_string.startswith('MEDICAL//'):
                code = token_string.split('//')[1].upper()
                return f"<EVENT> {self.medical_lookup.get(code, code.replace('_', ' ').title())}"
            elif token_string.startswith('MEASUREMENT//'):
                code = token_string.split('//')[1].upper()
                description = self.medical_lookup.get(code, code.replace('_', ' ').title())
                return f"<EVENT> {description}"
            elif token_string.startswith('LIFESTYLE//'):
                code = token_string.split('//')[1].upper()
                return f"Lifestyle {code}"
            elif token_string.startswith('LAB//'):
                code = token_string.split('//')[1].upper()
                return f"<EVENT> {self.lab_lookup.get(code, code.replace('_', ' ').title())}"
            elif token_string.startswith(('GENDER//', 'ETHNICITY//')):
                parts = token_string.split('//')
                return f"<DEMOGRAPHIC> {parts[0]} {parts[1]}"
            elif token_string.startswith('REGION//'):
                parts = token_string.split('//')
                return f"<DEMOGRAPHIC> {parts[0]} {self.region_lookup.get(parts[1], parts[1])}"
            elif token_string.startswith('Q') and len(token_string) <= 4 and token_string[1:].isdigit():
                return f"<VALUE> {token_string[1:]}"
            elif token_string.startswith('low') or token_string.startswith('normal') or token_string.startswith('high') or token_string.startswith('very low') or token_string.startswith('very high') and len(token_string) <= 9:
                return f"<VALUE> {token_string}"
            elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return token_string
            else:
                return "Unknown"
        except Exception as e:
            print(f"Warning: Failed to translate '{token_string}': {e}")
            return ""
    
    def token_adaptation(self, model_name, max_seq_length=512, load_in_4bit=True):
        """
        Token adaptation pipeline that extends the tokenizer with predefined tokens.
        
        Args:
            model_name: Name of the model to load
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit precision
            
        Returns:
            tuple: (model, tokenizer) with extended vocabulary
        """
        
        
        # Define tokens to add to the tokenizer
        tokens_to_add = [
            "<TIME>", "<DEMOGRAPHIC>", "<EVENT>", "<VALUE>",
        ]
        
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        
        # Get current vocabulary
        current_vocab = tokenizer.get_vocab().keys()
        print(f"Current vocab size: {len(current_vocab)}")
        
        # Filter tokens that don't already exist
        new_tokens = []
        existing_tokens = []
        
        for token in tokens_to_add:
            if token not in current_vocab:
                new_tokens.append(token)
            else:
                existing_tokens.append(token)
        
        if existing_tokens:
            print(f'Warning: {len(existing_tokens)} tokens already exist in the tokenizer')
            print(f"Existing tokens: {existing_tokens}")
        
        # Add new tokens to tokenizer
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added} new tokens to tokenizer")
            print(f"New tokens: {new_tokens}")
            
            # Resize model embeddings to accommodate new tokens
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model embeddings to {len(tokenizer)} tokens")
        else:
            print("No new tokens to add")
        
        # Add PAD token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        print(f"Final vocab size: {len(tokenizer)}")
        
        return model, tokenizer