import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from method.step1_adv_sample_generation.CMRC2018.load_token_new import get_token

class SentencePairDataset(Dataset):
    def __init__(self, data, max_len, padding_idx):
        super(SentencePairDataset, self).__init__()

        self.data_size = len(data["input_ids"])
        self.sentences = []
        self.input_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.token_type_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.mask_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx

        self.start_positions = torch.ones((self.data_size, 1), dtype=torch.int64) * padding_idx
        # print(self.start_positions)
        self.end_positions = torch.ones((self.data_size, 1), dtype=torch.int64) * padding_idx

        text_length = list()

        for idx in range(self.data_size):
            content_len = min(len(data["input_ids"][idx]), max_len)
            self.sentences.append(''.join(data['tokens'][idx]))
            self.input_ids[idx][:content_len] = torch.tensor(data["input_ids"][idx][:content_len], dtype=torch.int64)
            self.token_type_ids[idx][:content_len] = torch.tensor(data["token_type_ids"][idx][:content_len], dtype=torch.int64)
            self.mask_ids[idx][:content_len] = torch.tensor(data["mask_ids"][idx][:content_len], dtype=torch.int64)
            text_length.append(content_len)
            # print(idx)
            # print(data["start_positions"])
            # print(data["start_positions"][idx])
            # print(torch.tensor(data["start_positions"][idx], dtype=torch.int64))
            self.start_positions[idx] = torch.tensor(data["start_positions"][idx], dtype=torch.int64)
            self.end_positions[idx] = torch.tensor(data["end_positions"][idx], dtype=torch.int64)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret_data = {
            "sentences": self.sentences[idx],
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "mask_ids": self.mask_ids[idx],
            "start_positions": self.start_positions[idx],
            "end_positions": self.end_positions[idx]
        }
        return ret_data

if __name__ == "__main__":
    pass


