import os
import pickle
import random
from logging import Logger
import pandas as pd
import torch
from tqdm import tqdm


class RolloutEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset for running evaluation of a Nightingale model. This class takes a directory (data_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and a rollout datapoint is attempted to be generated for each subject.
    This dataset is used for running evaluations where you want to see the models ability to predict an end token given a context window.

    Args:
        data_dir (str): The directory containing the pickled tokenized data.
        vocab_path (str): The path to the vocabulary file.
        sequence_length (int): The length of the input and target token sequences.
        start_token_id (int): The id of the start token.
        end_token_ids (list[int]): The ids of the end tokens.
        logger (Logger): The logger to use.
    """

    def __init__(self, dataset_dir: str, vocab_path: str, sequence_length: int, start_token_id: int = None, end_token_ids: list[int] = None, start_token_str: str = None, end_token_strs: list[str] = None, logger: Logger = None) -> None:
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.logger = logger
        self.vocab = pd.read_csv(vocab_path) # columns are token, str, count

        # set start token
        if start_token_id is not None:
            self.start_token_id = start_token_id
            self.start_token_str = self.vocab[self.vocab["token"] == start_token_id]["str"].values[0]
        elif start_token_str is not None:
            self.start_token_str = start_token_str
            self.start_token_id = self.vocab[self.vocab["str"] == start_token_str]["token"].values[0]
        else:
            raise ValueError("Either start_token_id or start_token_str must be provided")
        
        # set end tokens
        if end_token_ids is not None:
            self.end_token_ids = end_token_ids
            self.end_token_strs = [self.vocab[self.vocab["token"] == end_token_id]["str"].values[0] for end_token_id in end_token_ids]
        elif end_token_strs is not None:
            self.end_token_strs = end_token_strs
            self.end_token_ids = [self.vocab[self.vocab["str"] == end_token_str]["token"].values[0] for end_token_str in end_token_strs]
        else:
            raise ValueError("Either end_token_ids or end_token_strs must be provided")

        self.data = self.__load_data_from_dir__(dataset_dir)

    def __load_data_from_dir__(self, dataset_dir: str) -> list:
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
        ][:2]

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    
                    # try to generate a rollout datapoint
                    try:
                        input_tokens, input_timestamps, end_token, start_token_idx, end_token_idx = self.__generate_rollout__(subject_data["tokens"], subject_data["timestamps"])
                        data.append({
                            "subject_id": torch.tensor(subject_data["subject_id"]),
                            "input_tokens": torch.tensor(input_tokens),
                            "input_timestamps": torch.tensor(input_timestamps),
                            "end_token": torch.tensor(end_token),
                            "start_token_idx": torch.tensor(start_token_idx),
                            "end_token_idx": torch.tensor(end_token_idx)
                        })
                    except ValueError:
                        if self.logger is not None:
                            self.logger.warning(f"No rollout datapoint found for subject {subject_data['subject_id']}")
                        continue

        if self.logger is not None:
            self.logger.info(f"Loaded {len(data)} rollout datapoints from {dataset_dir}")
        return data
    
    def __generate_rollout__(self, tokens: list[int], timestamps: list[int]) -> list[int]:
        """
        Generates a rollout datapoint that starts from a start token and ends with an end token.

        Args:
            tokens (list[int]): The list of tokens to generate a rollout from.
            timestamps (list[int]): The list of timestamps to generate a rollout from.

        Returns:
            input_tokens (list[int]): The list of tokens to generate a rollout from.
            input_timestamps (list[int]): The list of timestamps to generate a rollout from.
            end_token (int): The end token that the rollout ends with.
            end_token_idx (int): The index of the end token that the rollout ends with.
        """

        # find the index of the last instance of the start token in the tokens list
        try:
            start_token_idx = len(tokens) - 1 - tokens[::-1].index(self.start_token_id)
        except ValueError:
            if self.logger is not None:
                self.logger.warning(f"Start token {self.start_token_id} not found in tokens")
            raise ValueError(f"Start token {self.start_token_id} not found in tokens")
        
        # if there are less than sequence_length tokens before the start token,
        # raise an error as there is not enough tokens for the context window
        # TODO: in future this should use padding and a mask so that these sample can be used for evaluation
        if len(tokens[:start_token_idx]) < self.sequence_length:
            if self.logger is not None:
                self.logger.warning(f"Less than {self.sequence_length} tokens before start token")
            raise ValueError(f"Less than {self.sequence_length} tokens before start token")
        
        # find the first instance of any of the end token ids that occur after the start token index
        for steps, token_id in enumerate(tokens[start_token_idx + 1:]):
            if token_id in self.end_token_ids:
                end_token_idx = steps + start_token_idx + 1
                end_token = token_id
                break
        else:
            # if no end token is found then return end_token = -1 and end_token_idx = -1
            end_token = -1
            end_token_idx = -1
        
        # create the input sequence so that the context window ends with the start token
        input_sequence_start_idx = start_token_idx - self.sequence_length + 1
        input_tokens = tokens[input_sequence_start_idx:start_token_idx + 1]
        input_timestamps = timestamps[input_sequence_start_idx:start_token_idx + 1]

        return input_tokens, input_timestamps, end_token, start_token_idx, end_token_idx
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
    

if __name__ == "__main__":

    dataset_parent_dir = "/home/joshua/data/mimic/mimic_iv/meds/mimic_iv_meds/tokenized_data/ethos_timetokens"
    dataset_dir = os.path.join(dataset_parent_dir, "tuning")
    vocab_path = os.path.join(dataset_parent_dir, "vocab.csv")

    start_token_str = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"

    end_token_strs = ["MEDS_DEATH", "TRANSFER_TO//discharge//UNKNOWN", "HOSPITAL_DISCHARGE//HOME", "HOSPITAL_DISCHARGE//UNK"]

    dataset = RolloutEvaluationDataset(dataset_dir, vocab_path, sequence_length=128, start_token_str=start_token_str, end_token_strs=end_token_strs, logger=None)

    print(dataset.start_token_id)
    print(dataset.start_token_str)
    print(dataset.end_token_ids)
    print(dataset.end_token_strs)

    for x in dataset:
        print(x['input_tokens'])
        print(x['end_token'])
        print(dataset.vocab[dataset.vocab["token"] == int(x['end_token'])]["str"].values[0])
        print(x['end_token_idx'])
        input()