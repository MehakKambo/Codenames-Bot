from torch.utils.data.dataset import Dataset
import torch

class GameboardDataset(Dataset):
    def __init__(self, generator):
        self.generator = generator

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        board, ourwords = self.generator.generateRandomBoard()
        # we need to replace all the indexes here with word encodings
        # we need to use the dictionary we created in main.py
        return torch.from_numpy(board), torch.from_numpy(ourwords)