import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, config):
        self.encodings = encodings
        self.config = config

    def __getitem__(self, idx):
        return {k: v[idx].to(self.config.device) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)