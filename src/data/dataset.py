import torch
import os
import pickle
import random
from tqdm import tqdm
import pandas as pd
from logging import Logger

class NightingaleTrainingDataset(torch.utils.data.Dataset):
    """
    Dataset for running training of a Nightingale model. This class takes a directory (data_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and added to self.data if it meets the criteria (defined in __load_data_from_dir__).

    Args:
        data_dir (str): The directory containing the pickled tokenized data.
        mode (str): The mode of the dataset, changes how data is loaded and returned. Must be one of "train", "eval".
        sequence_length (int): The length of the input and target token sequences.
    """
    def __init__(self, data_dir: str, mode: str, sequence_length: int = 100, logger: Logger = None) -> None:
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.logger = logger

        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'train', 'eval'.")

        # Populate self.data
        self.data = self.__load_data_from_dir__(data_dir)

    def __load_data_from_dir__(self, data_dir: str) -> list:
        """
        Loads data from the data directory and populates self.data depending on self.mode.

        Args:
            data_dir (str): The directory containing the pickled tokenized data.

        Returns:
            data (list): A list of dictionaries containing the data for each sample.
        """

        data = []

        # get all the pickle files in the data directory
        file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")]

        self.logger.info(f"Loading {len(file_paths)} files from {data_dir} for {self.mode} dataset")
        
        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    
                    # if the token sequence is less than the sequence length + 1, ignore this sample as not enough data
                    if len(subject_data['tokens']) < self.sequence_length + 1:
                        continue

                    # if this is a training dataset, then we append the whole token sequence and
                    # randomly sample a 'start index' in __getitem__ to create a token sequence of length sequence_length.
                    if self.mode == "train":

                        data.append({
                            'subject_id': torch.tensor(subject_data['subject_id']),
                            'tokens': torch.tensor(subject_data['tokens']),
                            'timestamps': torch.tensor(subject_data['timestamps'])
                        })

                    # if this is an evaluation dataset, then we preprocess the token sequence and chunk it into
                    # chunks of length sequence_length and append each chunk to self.data. This is done to have
                    # a deterministic dataset for evaluation, rather than randomly sampling a start index as in training.
                    if self.mode == "eval":
                        
                        chunks = []
                        for i in range(0, len(subject_data['tokens']) - self.sequence_length - 1, self.sequence_length):
                            chunks.append({
                                'subject_id': torch.tensor(subject_data['subject_id']),
                                'tokens': torch.tensor(subject_data['tokens'][i:i + self.sequence_length]),
                                'timestamps': torch.tensor(subject_data['timestamps'][i:i + self.sequence_length])
                            })

                        data.extend(chunks)

        self.logger.info(f"Loaded {len(data)} samples from {data_dir} for {self.mode} dataset")

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        x = self.data[idx]

        if self.mode == "train":
            # If this is a training dataset then we randomly sample a start index and construct the input and target token sequences.
        
            # select random start index
            start_idx = random.randint(0, len(x['tokens']) - self.sequence_length - 1)

            # construct input and target token sequences
            input_tokens = x['tokens'][start_idx:start_idx + self.sequence_length]
            input_timestamps = x['timestamps'][start_idx:start_idx + self.sequence_length]

            target_tokens = x['tokens'][start_idx + 1:start_idx + self.sequence_length + 1]
            target_timestamps = x['timestamps'][start_idx + 1:start_idx + self.sequence_length + 1]

        elif self.mode == "eval":
            # If this is an evaluation dataset then the sample has been preprocessed to be a chunk of length sequence_length.
            # We can just return the input and target token sequences as they are.
            input_tokens = x['tokens'][:-1]
            input_timestamps = x['timestamps'][:-1]
            target_tokens = x['tokens'][1:]
            target_timestamps = x['timestamps'][1:]

        return {
            'subject_id': x['subject_id'],
            'input_tokens': input_tokens,
            'input_timestamps': input_timestamps,
            'target_tokens': target_tokens,
            'target_timestamps': target_timestamps
        }

class NightingaleEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset for running evaluation of a Nightingale model. This class takes a directory (data_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and added to self.data if it meets the criteria (defined in __load_data_from_dir__).

    Args:
        data_dir (str): The directory containing the pickled tokenized data.
    """
    def __init__(self, data_dir: str, vocab_path: str, logger: Logger = None) -> None:
        self.data_dir = data_dir

        # Populate self.data
        self.data = self.__load_data_from_dir__(data_dir)
        self.subject_id_map = {int(subject_id): idx for idx, subject_id in enumerate(data['subject_id'] for data in self.data)}

        # Load vocab
        self.vocab = pd.read_csv(vocab_path, index_col=False)

    def __load_data_from_dir__(self, data_dir: str) -> list:
        """
        Loads data from the data directory and populates self.data.

        Args:
            data_dir (str): The directory containing the pickled tokenized data.

        Returns:
            data (list): A list of dictionaries containing the data for each sample.
        """

        data = []

        # get all the pickle files in the data directory
        file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")][:2]

        self.logger.info(f"Loading {len(file_paths)} files from {data_dir} for evaluation dataset")

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    data.append({
                        'subject_id': torch.tensor(subject_data['subject_id']),
                        'tokens': torch.tensor(subject_data['tokens']),
                        'timestamps': torch.tensor(subject_data['timestamps'])
                    })
        
        self.logger.info(f"Loaded {len(data)} samples from {data_dir} for evaluation dataset")
        
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
            return self.vocab.iloc[token]['str']
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
            return int(self.vocab[self.vocab['str'] == string]['token'].values[0])
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
        return [self.vocab.iloc[token]['str'] for token in tokens.tolist()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":

    dataset_dir = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/Template Tokenization Pipeline/tuning"
    # dataset = NightingaleTrainingDataset(dataset_dir, mode="train", sequence_length=100)
    dataset = NightingaleEvaluationDataset(dataset_dir)

    data = dataset.get_data_by_subject_id(10032725)
    print(data)
    exit()
    
    for datapoint in dataset:
        print(datapoint)
        input()