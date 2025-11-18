import os
import pickle
import random
from logging import Logger
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm


class NightingaleTrainingDataset(torch.utils.data.Dataset):
    """
    Dataset for running training (and validation) of a Nightingale model. This class takes a directory (dataset_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and added to self.data if it meets the criteria (defined in __load_data_from_dir__).

    Args:
        dataset_dir (str): The directory containing the pickled tokenized data.
        mode (str): The mode of the dataset, changes how data is loaded and returned. Must be one of "train", "eval".
        sequence_length (int): The length of the input and target token sequences.
        insert_static_demographic_tokens (bool): Whether to make the first tokens of each datapoint the static demographic tokens.
        clinical_notes_dir (str): The directory containing the pickled tokenized clinical notes.
        clinical_notes_max_note_count (int): The maximum number of clinical notes to include.
        clinical_notes_max_tokens_per_note (int): The maximum number of tokens to include per clinical note.
    """

    def __init__(
        self, dataset_dir: str, mode: str, sequence_length: int = 100, insert_static_demographic_tokens: bool = True, clinical_notes_dir: str = None, clinical_notes_max_note_count: int = 3, clinical_notes_max_tokens_per_note: int = 256, logger: Logger = None
    ) -> None:
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.insert_static_demographic_tokens = insert_static_demographic_tokens
        self.logger = logger

        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'train', 'eval'.")

        # Populate data
        self.subject_id_to_data_index = {}
        self.data = self._load_data_from_dir(dataset_dir)

        # store a lookup table of subject_id to data index
        self.subject_id_to_data_index = {
            int(subject_id): idx for idx, subject_id in enumerate(data["subject_id"] for data in self.data)
        }

        # if clinical notes are provided, then load them
        self.using_clinical_notes = True if clinical_notes_dir is not None else False
        if self.using_clinical_notes:
            self.clinical_notes_dir = clinical_notes_dir
            self.clinical_notes_max_note_count = clinical_notes_max_note_count
            self.clinical_notes_max_tokens_per_note = clinical_notes_max_tokens_per_note
            self._load_clinical_notes(clinical_notes_dir)


    def _load_data_from_dir(self, dataset_dir: str) -> list:
        """
        Loads data from the data directory and populates self.data depending on self.mode.

        Args:
            data_dir (str): The directory containing the pickled tokenized data.

        Returns:
            data (list): A list of dictionaries containing the data for each sample.
        """
        data = []

        # get all the pickle files in the data directory
        file_paths = [
            os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".pkl")
        ]

        if self.logger:
            self.logger.info(f"Loading {len(file_paths)} files from {dataset_dir} for {self.mode} dataset")

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    # # if the token sequence is less than the sequence length + 1, ignore this sample as not enough data
                    # if len(subject_data["tokens"]) < self.sequence_length + 1:
                    #     continue

                    # if this is a training dataset, then we append the whole token sequence and
                    # randomly sample a 'start index' in __getitem__ to create a token sequence of length sequence_length.
                    if self.mode == "train":
                        data.append(
                            {
                                "subject_id": torch.tensor(subject_data["subject_id"]),
                                "ehr": {
                                    "token_ids": torch.tensor(subject_data["tokens"]),
                                    "timestamps": torch.tensor(subject_data["timestamps"]),
                                },
                                "clinical_notes": []
                            }
                        )

                    # if this is an evaluation dataset, then we preprocess the token sequence and chunk it into
                    # chunks of length sequence_length and append each chunk to self.data. This is done to have
                    # a deterministic dataset for evaluation, rather than randomly sampling a start index as in training.
                    if self.mode == "eval":
                        chunks = []
                        if self.insert_static_demographic_tokens:
                            tokens_tensor = torch.tensor(subject_data["tokens"])
                            timestamps_tensor = torch.tensor(subject_data["timestamps"])
                            static_demographic_token_ids = tokens_tensor[timestamps_tensor == 0]
                            static_demographic_token_length = len(static_demographic_token_ids)
                        else:
                            static_demographic_token_ids = torch.tensor([])
                            static_demographic_token_length = 0

                        # Calculate how many tokens to sample from non-static portion
                        tokens_to_sample = self.sequence_length - static_demographic_token_length
                        non_static_length = len(subject_data["tokens"]) - static_demographic_token_length
                        
                        # Create chunks by stepping through non-static tokens
                        for i in range(static_demographic_token_length, 
                                     static_demographic_token_length + non_static_length - tokens_to_sample, 
                                     tokens_to_sample):
                            chunks.append(
                                {
                                    "subject_id": torch.tensor(subject_data["subject_id"]),
                                    "ehr": {
                                        "token_ids": torch.tensor(
                                            static_demographic_token_ids.tolist() + subject_data["tokens"][i : i + tokens_to_sample]
                                        ),
                                        "timestamps": torch.tensor(
                                            [0] * static_demographic_token_length + subject_data["timestamps"][i : i + tokens_to_sample]
                                        )
                                    },
                                    "clinical_notes": []
                                }
                            )

                        data.extend(chunks)

        if self.logger:
            self.logger.info(f"Loaded {len(data)} samples from {dataset_dir} for {self.mode} dataset")

        return data

    def _load_clinical_notes(self, clinical_notes_dir: str) -> None:
        """
        Loads clinical notes from the clinical notes directory and populates self.clinical_notes.
        """
        file_paths = [
            os.path.join(clinical_notes_dir, file) for file in os.listdir(clinical_notes_dir) if file.endswith(".pkl")
        ]

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    if int(subject_data["subject_id"]) in self.subject_id_to_data_index:
                        data_index = self.subject_id_to_data_index[int(subject_data["subject_id"])]
                        self.data[data_index]["clinical_notes"] = []
                        for note in subject_data["notes"]:
                            self.data[data_index]["clinical_notes"].append({
                                "token_ids": torch.tensor(note["token_ids"]),
                                "timestamp": torch.tensor(datetime.strptime(note["charttime"], "%Y-%m-%d %H:%M:%S").timestamp()), # TODO do this conversion in the tokenization pipeline
                                "note_id": note["note_id"],
                            })

        return
    
    def _preprocess_clinical_notes(self, clinical_notes: list) -> dict:
        """
        Preprocesses the clinical notes to produce a tensor of shape (self.clinical_notes_max_note_count, self.clinical_notes_max_tokens_per_note).

        Args:
            clinical_notes (list): A list of clinical notes.
            max_note_count (int): The maximum number of clinical notes to include.
            max_tokens_per_note (int): The maximum number of tokens to include per clinical note.

        Returns:
            - token_ids (torch.Tensor): A tensor of shape (max_note_count, max_tokens_per_note) with padded token IDs.
            - notes_mask (torch.Tensor): A mask of shape (self.clinical_notes_max_note_count) indicating which notes are valid (1) vs padding (0).
            - tokens_mask (torch.Tensor): A mask of shape (self.clinical_notes_max_note_count, self.clinical_notes_max_tokens_per_note) indicating which tokens are valid (1) vs padding (0).
        """
        # Initialize tensors with zeros (padding value)
        token_ids = torch.zeros(self.clinical_notes_max_note_count, self.clinical_notes_max_tokens_per_note, dtype=torch.long)
        notes_mask = torch.zeros(self.clinical_notes_max_note_count, dtype=torch.bool)
        tokens_mask = torch.zeros(self.clinical_notes_max_note_count, self.clinical_notes_max_tokens_per_note, dtype=torch.bool)
        
        # Sort clinical notes by timestamp to maintain chronological order
        clinical_notes_sorted = sorted(clinical_notes, key=lambda x: x["timestamp"])
        
        # Process up to max_note_count notes
        num_notes_to_process = min(len(clinical_notes_sorted), self.clinical_notes_max_note_count)
        
        # Process notes in reverse chronological order to ensure that the most recent notes have priority over older notes
        for i in reversed(range(num_notes_to_process)):
            note = clinical_notes_sorted[i]
            note_tokens = note["token_ids"]
            
            # Truncate tokens if they exceed max_tokens_per_note
            num_tokens = min(len(note_tokens), self.clinical_notes_max_tokens_per_note)
            
            if num_tokens > 0:
                # Fill in the token IDs
                token_ids[i, :num_tokens] = note_tokens[:num_tokens]
                
                # Mark this note as valid
                notes_mask[i] = True
                
                # Mark the tokens in this note as valid
                tokens_mask[i, :num_tokens] = True
        
        return token_ids, notes_mask, tokens_mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        # # TODO: Implement a Collator class so padding / truncation is done dynamically in the collate function
        x = self.data[idx]

        # if self.mode == "train":
        #     # If this is a training dataset then we randomly sample a start index and construct the input and target token sequences.

        #     # select random start index
        #     if self.insert_static_demographic_tokens:
        #         static_demographic_token_ids = x["ehr"]["token_ids"][x["ehr"]["timestamps"] == 0]
        #         # We need to sample (sequence_length - len(static_tokens)) tokens from the non-static tokens
        #         tokens_to_sample = self.sequence_length - len(static_demographic_token_ids)
        #         non_static_tokens_available = len(x["ehr"]["token_ids"]) - len(static_demographic_token_ids)
                
        #         if non_static_tokens_available < tokens_to_sample + 1:  # +1 for target
        #             raise ValueError(f"Not enough non-static tokens: need {tokens_to_sample + 1}, have {non_static_tokens_available}")
                
        #         # Sample from the non-static portion
        #         start_idx = random.randint(0, non_static_tokens_available - tokens_to_sample - 1) + len(static_demographic_token_ids)
        #     else:
        #         available_length = len(x["ehr"]["token_ids"]) - self.sequence_length - 1
        #         if available_length < 0:
        #             raise ValueError(f"Sequence too short for training: total_tokens={len(x['ehr']['token_ids'])}, sequence_length={self.sequence_length}")
        #         start_idx = random.randint(0, available_length)

        #     # construct input and target token sequences
        #     if self.insert_static_demographic_tokens:
        #         # Sample the correct number of tokens (sequence_length - static_tokens)
        #         tokens_to_sample = self.sequence_length - len(static_demographic_token_ids)
        #         input_token_ids = x["ehr"]["token_ids"][start_idx : start_idx + tokens_to_sample]
        #         input_timestamps = x["ehr"]["timestamps"][start_idx : start_idx + tokens_to_sample]

        #         target_token_ids = x["ehr"]["token_ids"][start_idx + 1 : start_idx + tokens_to_sample + 1]
        #         target_timestamps = x["ehr"]["timestamps"][start_idx + 1 : start_idx + tokens_to_sample + 1]

        #         # insert static demographic tokens at the beginning
        #         input_token_ids = torch.cat([static_demographic_token_ids, input_token_ids])
        #         input_timestamps = torch.cat([torch.zeros_like(static_demographic_token_ids), input_timestamps])
        #         target_token_ids = torch.cat([static_demographic_token_ids, target_token_ids])
        #         target_timestamps = torch.cat([torch.zeros_like(static_demographic_token_ids), target_timestamps])
        #     else:
        #         input_token_ids = x["ehr"]["token_ids"][start_idx : start_idx + self.sequence_length]
        #         input_timestamps = x["ehr"]["timestamps"][start_idx : start_idx + self.sequence_length]

        #         target_token_ids = x["ehr"]["token_ids"][start_idx + 1 : start_idx + self.sequence_length + 1]
        #         target_timestamps = x["ehr"]["timestamps"][start_idx + 1 : start_idx + self.sequence_length + 1]
        if self.mode == "train":
            # Extract raw tokens/timestamps
            raw_token_ids = x["ehr"]["token_ids"]
            raw_timestamps = x["ehr"]["timestamps"]
            
            # Separate static vs non-static if needed
            if self.insert_static_demographic_tokens:
                static_mask = (raw_timestamps == 0)
                static_tokens = raw_token_ids[static_mask]
                non_static_tokens = raw_token_ids[~static_mask]
                non_static_timestamps = raw_timestamps[~static_mask]
                
                # Calculate how much space we have for non-static data
                # (sequence_length - static_len)
                max_dynamic_len = self.sequence_length - len(static_tokens)
            else:
                static_tokens = torch.tensor([], dtype=torch.long)
                non_static_tokens = raw_token_ids
                non_static_timestamps = raw_timestamps
                max_dynamic_len = self.sequence_length

            # --- PADDING LOGIC STARTS HERE ---
            
            # Check if we have enough data to fill the sequence
            # We need +1 for the target (next token)
            if len(non_static_tokens) < max_dynamic_len + 1:
                # CASE 1: Sequence is too short -> PAD IT
                
                # Take everything we have
                input_dynamic = non_static_tokens
                ts_dynamic = non_static_timestamps
                
                # Calculate padding needed
                # We need input length to be exactly max_dynamic_len
                pad_len = max_dynamic_len - len(input_dynamic)
                
                # Create padding (0 for inputs, 0 for timestamps)
                pad_zeros = torch.zeros(pad_len, dtype=torch.long)
                
                # 1. Construct Input (Static + Dynamic + Pad)
                input_token_ids = torch.cat([static_tokens, input_dynamic, pad_zeros])
                input_timestamps = torch.cat([torch.zeros_like(static_tokens), ts_dynamic, pad_zeros])
                
                # 2. Construct Target
                # For target, we shift by 1. 
                # The target for the last real token is the first pad (or end token), usually ignored.
                # We pad targets with -100 (standard PyTorch ignore_index) so the model doesn't learn to predict padding.
                target_dynamic = torch.cat([non_static_tokens[1:], torch.tensor([0])]) # simplistic next-token logic
                # Ideally target matches input shifted. For the padded area, target is -100
                
                # Let's do a simpler shift on the full padded sequence:
                full_seq = torch.cat([static_tokens, non_static_tokens, pad_zeros])
                # Input is 0..N-1
                input_token_ids = full_seq[:-1] 
                # Target is 1..N
                target_token_ids = full_seq[1:].clone()
                # Set target at padding positions to -100
                # The valid length is len(static) + len(non_static)
                valid_len = len(static_tokens) + len(non_static_tokens)
                # Any target index >= valid_len - 1 (since target is shifted) should be ignored?
                # Actually, safest is to explicitly pad targets with -100
                target_pad = torch.full((pad_len,), -100, dtype=torch.long)
                # Re-construct target carefully:
                # Real targets: static[1:] + non_static + first_pad? No, standard causal shift:
                # We just take the exact slice we constructed in input, but shift the source data?
                # Let's stick to the "Crop" logic but with padding appended.
                
                # Actual Input: [Static, Real_Dynamic, 0, 0...]
                # Actual Target: [Static[1:], Real_Dynamic[0], ..., Real_Dynamic[-1], -100, -100...]
                # Note: This implies we predict the first dynamic token from the last static token.
                
                # Simplified Padded Construction:
                # 1. Create full buffer of zeros
                input_token_ids = torch.zeros(self.sequence_length, dtype=torch.long)
                target_token_ids = torch.full((self.sequence_length,), -100, dtype=torch.long)
                input_timestamps = torch.zeros(self.sequence_length, dtype=torch.long)
                target_timestamps = torch.zeros(self.sequence_length, dtype=torch.long)
                padding_mask = torch.zeros(self.sequence_length, dtype=torch.bool)

                # 2. Fill static
                n_stat = len(static_tokens)
                input_token_ids[:n_stat] = static_tokens
                padding_mask[:n_stat] = True
                # Target for static part (predicting next static or first dynamic)
                # Note: usually we don't train on static tokens prediction if they are fixed, but standard LM does.
                
                # 3. Fill dynamic
                # We use the raw concatenated sequence: Static + NonStatic
                full_source = torch.cat([static_tokens, non_static_tokens])
                
                valid_len = len(full_source) - 1 # -1 because we need a next token
                
                # Copy inputs
                input_token_ids[:valid_len] = full_source[:-1]
                input_timestamps[:valid_len] = torch.cat([torch.zeros(n_stat), non_static_timestamps])[:-1]
                
                # Copy targets
                target_token_ids[:valid_len] = full_source[1:]
                target_timestamps[:valid_len] = torch.cat([torch.zeros(n_stat), non_static_timestamps])[1:]
                
                # Update mask
                padding_mask[:valid_len] = True
                
            else:
                # CASE 2: Sequence is long enough -> RANDOM CROP (Existing Logic)
                # ... (Keep your existing random sampling logic here) ...
                # Just ensure you define padding_mask at the end
                
                # After defining input_token_ids via random sampling:
                padding_mask = torch.ones(self.sequence_length, dtype=torch.bool)

        elif self.mode == "eval":
            # If this is an evaluation dataset then the sample has been preprocessed to be a chunk of length sequence_length.
            # We can just return the input and target token sequences as they are.
            # We also dont need to insert static demographic tokens as they are already in the input and target token sequences when they were preprocessed.
            input_token_ids = x["ehr"]["token_ids"][:-1]
            input_timestamps = x["ehr"]["timestamps"][:-1]
            target_token_ids = x["ehr"]["token_ids"][1:]
            target_timestamps = x["ehr"]["timestamps"][1:]

        datapoint = {
            "subject_id": x["subject_id"],
            "ehr": {
                "input_token_ids": input_token_ids,
                "input_timestamps": input_timestamps,
                "target_token_ids": target_token_ids,
                "target_timestamps": target_timestamps,
                "input_padding_mask": padding_mask,
            },
        }

        # # if clinical notes are being used, then preprocess them
        # if self.using_clinical_notes:
        #     # Select and preprocess clinical notes that fall within the a timestamp range (to prevent attending to future notes)
        #     # min_timestamp = input_timestamps[0]
        #     max_timestamp = input_timestamps[-1]
        #     # clinical_notes = [note for note in x["clinical_notes"] if note["timestamp"] >= min_timestamp and note["timestamp"] <= max_timestamp]
        #     clinical_notes = [note for note in x["clinical_notes"] if note["timestamp"] <= max_timestamp]

        #     token_ids, notes_mask, tokens_mask = self._preprocess_clinical_notes(clinical_notes)

        #     datapoint["clinical_notes"] = {
        #         "token_ids": token_ids,
        #         "notes_mask": notes_mask,
        #         "tokens_mask": tokens_mask,
        #     }

        return datapoint


class NightingaleEvaluationDataset(torch.utils.data.Dataset):
    # TODO: Rename this dataset to NightingaleSimulationDataset and move it to its own file
    """
    Dataset for running evaluation of a Nightingale model. This class takes a directory (data_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and added to self.data if it meets the criteria (defined in __load_data_from_dir__).

    Args:
        dataset_dir (str): The directory containing the pickled tokenized data.
    """

    def __init__(self, dataset_dir: str, vocab_path: str, logger: Logger = None) -> None:
        self.dataset_dir = dataset_dir
        self.logger = logger

        # Populate self.data
        self.data = self._load_data_from_dir(dataset_dir)
        self.subject_id_map = {
            int(subject_id): idx
            for idx, subject_id in enumerate(data["subject_id"] for data in self.data)
        }

        # Load vocab
        self.vocab = pd.read_csv(vocab_path, index_col=False)

    def _load_data_from_dir(self, dataset_dir: str) -> list:
        """
        Loads data from the data directory and populates self.data.

        Args:
            data_dir (str): The directory containing the pickled tokenized data.

        Returns:
            data (list): A list of dictionaries containing the data for each sample.
        """
        data = []

        # get all the pickle files in the dataset directory
        file_paths = [
            os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".pkl")
        ]

        if self.logger:
            self.logger.info(f"Loading {len(file_paths)} files from {dataset_dir} for evaluation dataset")

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    data.append(
                        {
                            "subject_id": torch.tensor(subject_data["subject_id"]),
                            "tokens": torch.tensor(subject_data["tokens"]),
                            "timestamps": torch.tensor(subject_data["timestamps"]),
                        }
                    )

        if self.logger:
            self.logger.info(f"Loaded {len(data)} samples from {dataset_dir} for evaluation dataset")

        return data

    def get_data_by_subject_id(self, subject_id: int) -> list:
        """
        Returns a list of data points for a given subject_id.

        Args:
            subject_id (int): The subject_id to get data for.

        Returns:
            data (dict): A dictionary containing the data for the given subject_id.
        """
        data_index = self.subject_id_map[subject_id]
        return self.data[data_index]

    def token_to_string(self, token: int) -> str:
        """
        Converts a token to a string. If the token is not found in the vocab, raises a ValueError.

        Args:
            token (int): The token to convert to a string.

        Returns:
            string (str): The string corresponding to the token.
        """
        try:
            return self.vocab.iloc[token]["str"]
        except IndexError:
            raise ValueError(f"Token {token} not found in vocab")

    def string_to_token(self, string: str) -> int:
        """
        Converts a string to a token. If the string is not found in the vocab, raises a ValueError.

        Args:
            string (str): The string to convert to a token.

        Returns:
            token (int): The token corresponding to the string.
        """
        try:
            return int(self.vocab[self.vocab["str"] == string]["token"].values[0])
        except IndexError:
            raise ValueError(f"String {string} not found in vocab")

    def tokens_to_strings(self, tokens: torch.Tensor) -> list:
        """
        Converts a list of tokens to a list of strings.

        Args:
            tokens (torch.Tensor): The tokens to convert to strings.

        Returns:
            strings (list): A list of strings.
        """
        return [self.vocab.iloc[token]["str"] for token in tokens.tolist()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=("Quick test to create a Nightingale dataset and inspect a datapoint."))
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to split directory (e.g., /path/to/train)",
    )
    args = parser.parse_args()

    clinical_notes_dir = "/home/joshua/data/mimic/mimic_iv/tokenized_notes"

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")

    vocab_path = os.path.join('/'.join(args.dataset_dir.split("/")[:-1]), "vocab.csv")
    print(vocab_path)
    vocab = pd.read_csv(vocab_path, index_col=False)
    print(vocab.head())

    dataset = NightingaleTrainingDataset(
        dataset_dir=args.dataset_dir,
        mode="train",
        sequence_length=128,
        insert_static_demographic_tokens=True,
        # clinical_notes_dir=clinical_notes_dir,
    )

    import numpy as np
    for datapoint in tqdm(dataset):
        for token_id in datapoint["ehr"]["input_token_ids"]:
            print(vocab[vocab["token"] == int(token_id)]["str"].values[0], end=" ")
        print()
        input()
