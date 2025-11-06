import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unsloth
from unsloth import FastLanguageModel
import torch.nn.functional as F
import pandas as pd


class EHRTokenTranslator:
    """
    Class to handle translation of EHR tokens to natural language.
    Uses the same translation logic as UnifiedEHRDataset.
    """
    
    def __init__(self, medical_lookup_filepath, lab_lookup_filepath):
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
    
    def _translate_token(self, token_string):
        """
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
                return ""
            elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return ""
            else:
                return ""
        except Exception as e:
            print(f"Warning: Failed to translate '{token_string}': {e}")
            return ""
    
    def extract_translated_concepts(self, vocab_filepath):
        """
        Extract all unique translated concepts from the EHR vocabulary.
        
        Args:
            vocab_filepath: Path to vocab.csv (columns: token, str)
            
        Returns:
            list: Unique list of natural language concepts
        """
        # Load vocab
        vocab_df = pd.read_csv(vocab_filepath)
        
        token_strings = vocab_df['str'].values 
        
        translated_concepts = []
        
        for token_string in token_strings:
            
            # If it's a measurable concept, add all 3 binned variations
            if token_string.startswith('LAB//') or token_string.startswith('MEASUREMENT//'):
                concept = self._translate_token(token_string) # e.g., "HbA1c"
                if concept:
                    translated_concepts.append(f"{concept}: Low")
                    translated_concepts.append(f"{concept}: Normal")
                    translated_concepts.append(f"{concept}: High")
            if token_string.startswith('AGE_') or token_string.startswith('BMI//') or token_string.startswith('HEIGHT//') or token_string.startswith('WEIGHT//'):
                concept = self._translate_token(token_string)
                if concept:
                    translated_concepts.append(f"{concept} Low")
                    translated_concepts.append(f"{concept} Normal")
                    translated_concepts.append(f"{concept} High")

            
            # If it's a quantile token, skip it
            elif token_string.startswith('Q') and token_string[1:].isdigit():
                continue
            
            # Otherwise, just translate and add the single token
            else:
                phrase = self._translate_token(token_string)
                if phrase: # Add if not an empty string
                    translated_concepts.append(phrase)
        
        # Return unique concepts
        unique_concepts = sorted(list(set(translated_concepts)))
        print(f"Extracted {len(unique_concepts)} unique concepts from vocabulary")
        print(f"Total vocabulary size: {len(vocab_df)}")

        for concept in unique_concepts[:10]:
            print(f"{concept}")
        
        return unique_concepts
    

    def token_adaptation(self,original_model_name, unsloth_model_name, new_concepts, max_seq_length=512, load_in_4bit=True):
        '''
        Token adaptation pipeline for embedding-based models.
        Args:
            original_model_name: Name of the original model.
            unsloth_model_name: Name of the unsloth model.
            new_concepts: List of new concepts to add to the model.
        Returns:
            model: The adapted model.
        '''
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(unsloth_model_name)

        # Check for original tokens
        current_vocab = tokenizer.get_vocab().keys()
        tokens_to_add = []
        tokens_that_exist = []

        for concept in new_concepts:
            if concept not in current_vocab:
                tokens_to_add.append(concept)
            else:
                tokens_that_exist.append(concept)

        if tokens_that_exist:
            print(f'Warning: {len(tokens_that_exist)} tokens already exist in the tokenizer')
            print(f"Tokens that already exist: {tokens_that_exist}")

        # Add new tokens
        num_new_tokens = tokenizer.add_tokens(tokens_to_add)

        # Add PAD token if needed
        pad_token_was_added = False
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_token_was_added = True
            print(f"PAD token was added to the tokenizer: {tokenizer.pad_token}")
        else:
            print(f"PAD token already exists in the tokenizer: {tokenizer.pad_token}")

        print(f"Added {num_new_tokens} new tokens. New vocab size = {len(tokenizer)}")

        # Load the model FIRST (without modifying vocab_size)
        model, _ = FastLanguageModel.from_pretrained(
            unsloth_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        print("\n--- DEBUG: Checking Model Layer Dimensions ---")
        input_embeds_before = model.get_input_embeddings()
        output_embeds_before = model.get_output_embeddings()
        print(f"Input embeddings BEFORE resize:  {input_embeds_before.weight.shape}")
        print(f"Output embeddings BEFORE resize: {output_embeds_before.weight.shape}")

        # Now resize embeddings to match the tokenizer
        model.resize_token_embeddings(len(tokenizer))

        input_embeds_after = model.get_input_embeddings()
        output_embeds_after = model.get_output_embeddings()
        print(f"Input embeddings AFTER resize:   {input_embeds_after.weight.shape}")
        print(f"Output embeddings AFTER resize:  {output_embeds_after.weight.shape}")

        # Initialize the new embeddings (average of sub-token embeddings)
        original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            original_model_name,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
        original_weights = base_model.get_input_embeddings().weight.data.cpu().float()
        del base_model

        new_input_embeddings = model.get_input_embeddings().weight.data
        new_output_embeddings = model.get_output_embeddings().weight.data

        print("Initializing new token weights in INPUT and OUTPUT layers...")
        with torch.no_grad():
            for concept in tokens_to_add:
                new_token_id = tokenizer.convert_tokens_to_ids(concept)
                sub_token_ids = original_tokenizer.encode(concept, add_special_tokens=False)
                
                # if not sub_token_ids:
                #     print(f"  Warning: Concept '{concept}' has no sub-tokens. Initializing as zero.")
                #     avg_emb = torch.zeros(original_weights.shape[1])
                # else:
                sub_embs = original_weights[sub_token_ids]
                avg_emb = sub_embs.mean(dim=0)

                # Assign to both input and output layers
                avg_emb_typed = avg_emb.to(new_input_embeddings.device, dtype=new_input_embeddings.dtype)
                new_input_embeddings[new_token_id] = avg_emb_typed
                new_output_embeddings[new_token_id] = avg_emb_typed

        print(f'Length of tokenizer: {len(tokenizer)}')
        print(f'Length of input embeddings: {model.get_input_embeddings().weight.data.shape[0]}')
        print(f'Length of output embeddings: {model.get_output_embeddings().weight.data.shape[0]}')


        print("\n--- DEBUG: Verifying Initialization (Checking one token) ---")
        if tokens_to_add:
            test_token_str = tokens_to_add[0]
            test_token_id = tokenizer.convert_tokens_to_ids(test_token_str)
            print(f"Checking token: '{test_token_str}' (ID: {test_token_id})")
            
            input_w = new_input_embeddings[test_token_id]
            output_w = new_output_embeddings[test_token_id]
            
            print(f"  Input Embed (mean):  {input_w.mean().item():.4f}")
            print(f"  Output Embed (mean): {output_w.mean().item():.4f}")
            
            if input_w.mean().item() == 0:
                print("  WARNING: Test token embedding is all zeros. Check `original_tokenizer.encode` step.")
            else:
                print("  SUCCESS: Test token is non-zero.")
        else:
            print("  No new tokens were added, skipping verification.")
        print("--------------------------------------------------------\n")
        # For checking the cosine similarity of the new embeddings to the original embeddings
        # embedding_weights = model.get_input_embeddings().weight.data
        # for concept in tokens_to_add:
        #     new_id = tokenizer.convert_tokens_to_ids(concept)
        #     sub_ids = original_tokenizer.encode(concept, add_special_tokens=False)
        #     sub_embs = original_weights[sub_ids]
        #     avg_emb = sub_embs.mean(dim=0).to(embedding_weights.device)
        #     new_emb = embedding_weights[new_id]
        #     print(f"{concept}: mean={new_emb.mean():.4f}, std={new_emb.std():.4f}")
        #     cos_sim = F.cosine_similarity(new_emb.unsqueeze(0), avg_emb.unsqueeze(0)).item()
        #     print(f"{concept}: cosine similarity to averaged sub-embedding = {cos_sim:.4f}")
        #     print(max(min(cos_sim, 1.0), -1.0))

        return model, tokenizer

if __name__ == "__main__":
    original_model_name = "Qwen/Qwen3-0.6B"
    unsloth_model_name = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    new_concepts = ["myocardial infarction", "Current smoker", "Bp diastolic", "7"]
    medical_lookup_filepath = "/data/home/qc25022/cancer-extraction-pipeline/src/resources/MedicalDictTranslation.csv"
    lab_lookup_filepath = "/data/home/qc25022/cancer-extraction-pipeline/src/resources/LabLookUP.csv"
    vocab_filepath = "/data/scratch/qc25022/upgi/tokenised_data_debug/cprd_test/vocab.csv"
    translator = EHRTokenTranslator(medical_lookup_filepath, lab_lookup_filepath)
    unique_concepts = translator.extract_translated_concepts(vocab_filepath)
    print(f"Unique concepts: {unique_concepts}")
    # model, tokenizer = translator.token_adaptation(original_model_name, unsloth_model_name, unique_concepts, max_seq_length=512, load_in_4bit=True)