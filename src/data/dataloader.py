from torch.utils.data import DataLoader
from .dataset import NightingaleDataset

def get_dataloader(dataset_dir: str, batch_size: int, shuffle: bool = True, sequence_length: int = 100, mode: str = "train"):
    """
    Creates a dataloader for a Nightingale dataset.

    Args:
        dataset_dir (str): The directory containing the pickled tokenized data.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.
        sequence_length (int): The length of the input and target token sequences.
        mode (str): The mode of the dataset, changes how data is loaded and returned. Must be one of "train", "eval".

    Returns:
        dataloader (DataLoader): A dataloader for the Nightingale dataset.
    """

    if mode not in ["train", "eval"]:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'train', 'eval'.")

    dataset = NightingaleDataset(dataset_dir, mode, sequence_length)
    return DataLoader(dataset, batch_size, shuffle)


if __name__ == "__main__":
    dataloader = get_dataloader("/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/Template Tokenization Pipeline/train", batch_size=10, shuffle=True)
    for batch in dataloader:
        print(batch['subject_id'])
        print(batch['tokens'].shape)
        input()