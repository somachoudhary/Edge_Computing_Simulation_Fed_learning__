import json
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, json_file):
        """
        Initialize dataset by loading JSON file.

        Args:
            json_file (str): Path to the dataset JSON file.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.samples = self.data["samples"]
        self.labels = self.data["labels"]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetches a single sample and its corresponding label."""
        sample = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

def get_dataloader(json_file, batch_size=32, shuffle=True):
    """
    Returns a DataLoader for the dataset.

    Args:
        json_file (str): Path to the JSON dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset = CustomDataset(json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example Usage
if __name__ == "__main__":
    json_path = "sample_dataset1.json"  # Ensure this file exists in the same directory
    dataloader = get_dataloader(json_path)

    for batch in dataloader:
        samples, labels = batch
        print("Sample Batch Shape:", samples.shape)
        print("Label Batch Shape:", labels.shape)
        break  # Print only first batch
