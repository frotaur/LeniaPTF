from torch.utils.data import Dataset
import os, torch

from tqdm import tqdm

class MemoryRewardDataset(Dataset):
    """
    Dataset for tensor preference annotations. Loads all data in memory.
    Memory loading very inefficient since refreshed for every new datapoint.
    To be used only for small amounts of data, and data which is light.

    Returns data as (tensor1, tensor2, annotation) tuples.
    Annotation is a float.
    """

    def __init__(self, data_folder):
        """
            Args:
            data_folder : str, folder containing the data as individual .pt files
        """
        self.data_folder = data_folder
        self.data_files = [os.path.join(data_folder,file) for file in os.listdir(data_folder) if file.endswith('.pt')]
        self.data = []

        self._load_in_memory()
        self.length = len(self.data_files)
    
    def _load_in_memory(self):
        print('memory loading')
        for file in tqdm(self.data_files):
            self.data.append(torch.load(file, map_location='cpu'))    

    def refresh(self):
        """
            Refreshes the data in memory. Useful if new data is added.
        """
        self.data_files = [os.path.join(self.data_folder,file) for file in os.listdir(self.data_folder) if file.endswith('.pt')]
        self.length = len(self.data_files)
        self._load_in_memory()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = self.data[idx]
        return data['data1'], data['data2'], data['annotation']