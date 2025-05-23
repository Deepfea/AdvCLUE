import os
import re
from transformers import BertTokenizer
import pandas as pd
import torch

class ClozeDataset:

    """
    Dataset which stores the tweets and returns them as processed features
    """

    def __init__(self, data_df, candidates, candidate_ids, path):
        # self.data_id = data_df.data_id.values
        self.tag = data_df.tag.values
        self.text = data_df.text.values
        self.candidate = candidates
        self.groundTruth = data_df.groundTruth.values
        self.labels = data_df.label.values
        self.tokenizer = BertTokenizer.from_pretrained(path, lowercase=True)
        self.max_len = 256
        self.candidate_ids = candidate_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # idiom_vocab = eval(
        #     open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
        # idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
        feature_id = int(self.tag[item][6: -1])
        # print(self.tag[item])
        # print(self.text[item])
        left_part, right_part = re.split(self.tag[item], self.text[item])
        left_ids = self.tokenizer.encode(left_part, add_special_tokens=False)
        right_ids = self.tokenizer.encode(right_part, add_special_tokens=False)

        half_length = int(self.max_len / 2)
        if len(left_ids) < half_length:  # cut at tail
            st = 0
            ed = min(len(left_ids) + 1 + len(right_ids), self.max_len - 2)
        elif len(right_ids) < half_length:  # cut at head
            ed = len(left_ids) + 1 + len(right_ids)
            st = max(0, ed - (self.max_len - 2))
        else:  # cut at both sides
            st = len(left_ids) + 3 - half_length
            ed = len(left_ids) + 1 + half_length

        text_ids = left_ids + [self.tokenizer.mask_token_id] + right_ids
        input_ids = [self.tokenizer.cls_token_id] + text_ids[st:ed] + [self.tokenizer.sep_token_id]

        position = input_ids.index(self.tokenizer.mask_token_id)

        token_type_ids = [0] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_masks = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        # print(self.candidate[item])
        # print(self.groundTruth[item])

        label = int(self.labels[item])

        # label = list(self.candidate[item]).index(self.groundTruth[item])
        # print(label)

        # print(idiom_vocab)
        # idiom_ids = [idiom_vocab[each] for each in self.candidate[item]]
        idiom_ids = list(self.candidate_ids[item])
        # print(idiom_ids)

        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        assert len(token_type_ids) == self.max_len

        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            # 'data_id': torch.tensor(self.data_id[item], dtype=torch.long),
            'feature_id': torch.tensor(feature_id, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_masks': torch.tensor(input_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'idiom_ids': torch.tensor(idiom_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'position': torch.tensor(position, dtype=torch.long)
        }

class ori_ClozeDataset:

    """
    Dataset which stores the tweets and returns them as processed features
    """

    def __init__(self, data_df, candidates, candidate_ids, path):
        # self.data_id = data_df.data_id.values
        self.tag = data_df.tag.values
        self.text = data_df.ori_text.values
        self.candidate = candidates
        self.groundTruth = data_df.groundTruth.values
        self.labels = data_df.label.values
        self.tokenizer = BertTokenizer.from_pretrained(path, lowercase=True)
        self.max_len = 256
        self.candidate_ids = candidate_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # idiom_vocab = eval(
        #     open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
        # idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
        feature_id = int(self.tag[item][6: -1])
        # print(self.tag[item])
        # print(self.text[item])
        left_part, right_part = re.split(self.tag[item], self.text[item])
        left_ids = self.tokenizer.encode(left_part, add_special_tokens=False)
        right_ids = self.tokenizer.encode(right_part, add_special_tokens=False)

        half_length = int(self.max_len / 2)
        if len(left_ids) < half_length:  # cut at tail
            st = 0
            ed = min(len(left_ids) + 1 + len(right_ids), self.max_len - 2)
        elif len(right_ids) < half_length:  # cut at head
            ed = len(left_ids) + 1 + len(right_ids)
            st = max(0, ed - (self.max_len - 2))
        else:  # cut at both sides
            st = len(left_ids) + 3 - half_length
            ed = len(left_ids) + 1 + half_length

        text_ids = left_ids + [self.tokenizer.mask_token_id] + right_ids
        input_ids = [self.tokenizer.cls_token_id] + text_ids[st:ed] + [self.tokenizer.sep_token_id]

        position = input_ids.index(self.tokenizer.mask_token_id)

        token_type_ids = [0] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_masks = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        # print(self.candidate[item])
        # print(self.groundTruth[item])

        label = int(self.labels[item])

        # label = list(self.candidate[item]).index(self.groundTruth[item])
        # print(label)

        # print(idiom_vocab)
        # idiom_ids = [idiom_vocab[each] for each in self.candidate[item]]
        idiom_ids = list(self.candidate_ids[item])
        # print(idiom_ids)

        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        assert len(token_type_ids) == self.max_len

        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            # 'data_id': torch.tensor(self.data_id[item], dtype=torch.long),
            'feature_id': torch.tensor(feature_id, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_masks': torch.tensor(input_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'idiom_ids': torch.tensor(idiom_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'position': torch.tensor(position, dtype=torch.long)
        }



class Config:
    max_len = 510

    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese'
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)

    # TOKENIZER = BertTokenizer.from_pretrained(os.path.join(BERT_PATH, 'vocab.txt'), lowercase=True)

if __name__ == "__main__":
    pass