from torch.utils.data import Dataset
import os, torch

from tqdm import tqdm

class DiskRewardDataset(Dataset):
    """
    Dataset for tensor preference annotations. Loads data from disk on the fly.
    Efficient memory use, but might be slow for huge datapoints/big batch sizes.
    TODO : See if there is a way to speed up the loading process.

    Returns data as (tensor1, tensor2, annotation) tuples.
    Annotation is a float.
    """

    def __init__(self, data_folder):
        """
            Args:
            data_folder : str, folder containing the data as individual .pt files
            device : str, device to load the data on. Default is 'cpu'.
        """
        self.data_folder = data_folder
        self.data_files = [os.path.join(data_folder,file) for file in os.listdir(data_folder) if file.endswith('.pt')]
        self.data = []

        self.length = len(self.data_files)
    

    def refresh(self):
        """
            Refreshes the data in memory. Useful if new data is added.
        """
        self.data_files = [os.path.join(self.data_folder,file) for file in os.listdir(self.data_folder) if file.endswith('.pt')]
        self.length = len(self.data_files)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data['data1'], data['data2'], data['annotation']