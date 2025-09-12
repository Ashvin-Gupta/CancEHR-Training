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
        clinical_notes_dir (str): The directory containing the pickled tokenized clinical notes.
        clinical_notes_max_note_count (int): The maximum number of clinical notes to include.
        clinical_notes_max_tokens_per_note (int): The maximum number of tokens to include per clinical note.
    """

    def __init__(
        self, dataset_dir: str, mode: str, sequence_length: int = 100, clinical_notes_dir: str = None, clinical_notes_max_note_count: int = 3, clinical_notes_max_tokens_per_note: int = 256, logger: Logger = None
    ) -> None:
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.mode = mode
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
                    # if the token sequence is less than the sequence length + 1, ignore this sample as not enough data
                    if len(subject_data["tokens"]) < self.sequence_length + 1:
                        continue

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
                        for i in range(0, len(subject_data["tokens"]) - self.sequence_length - 1, self.sequence_length):
                            chunks.append(
                                {
                                    "subject_id": torch.tensor(subject_data["subject_id"]),
                                    "ehr": {
                                        "token_ids": torch.tensor(
                                            subject_data["tokens"][i : i + self.sequence_length]
                                        ),
                                        "timestamps": torch.tensor(
                                            subject_data["timestamps"][i : i + self.sequence_length]
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
        # TODO: Implement a Collator class so padding / truncation is done dynamically in the collate function
        x = self.data[idx]

        if self.mode == "train":
            # If this is a training dataset then we randomly sample a start index and construct the input and target token sequences.

            # select random start index
            start_idx = random.randint(0, len(x["ehr"]["token_ids"]) - self.sequence_length - 1)

            # construct input and target token sequences
            input_token_ids = x["ehr"]["token_ids"][start_idx : start_idx + self.sequence_length]
            input_timestamps = x["ehr"]["timestamps"][start_idx : start_idx + self.sequence_length]

            target_token_ids = x["ehr"]["token_ids"][start_idx + 1 : start_idx + self.sequence_length + 1]
            target_timestamps = x["ehr"]["timestamps"][
                start_idx + 1 : start_idx + self.sequence_length + 1
            ]

        elif self.mode == "eval":
            # If this is an evaluation dataset then the sample has been preprocessed to be a chunk of length sequence_length.
            # We can just return the input and target token sequences as they are.
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
            },
        }

        # if clinical notes are being used, then preprocess them
        if self.using_clinical_notes:
            # Select and preprocess clinical notes that fall within the a timestamp range (to prevent attending to future notes)
            # min_timestamp = input_timestamps[0]
            max_timestamp = input_timestamps[-1]
            # clinical_notes = [note for note in x["clinical_notes"] if note["timestamp"] >= min_timestamp and note["timestamp"] <= max_timestamp]
            clinical_notes = [note for note in x["clinical_notes"] if note["timestamp"] <= max_timestamp]

            token_ids, notes_mask, tokens_mask = self._preprocess_clinical_notes(clinical_notes)

            datapoint["clinical_notes"] = {
                "token_ids": token_ids,
                "notes_mask": notes_mask,
                "tokens_mask": tokens_mask,
            }

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

    dataset = NightingaleTrainingDataset(
        dataset_dir=args.dataset_dir,
        mode="train",
        sequence_length=128,
        clinical_notes_dir=clinical_notes_dir,
    )

    import numpy as np
    for datapoint in tqdm(dataset, desc="Counting clinical notes"):
        print(datapoint)
        input()
