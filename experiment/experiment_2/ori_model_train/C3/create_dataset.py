
import json
import os
import random
from tqdm import tqdm

import pandas as pd
import torch

from torch.utils.data import TensorDataset

class InputExample(object):
    def __init__(self, text_a, text_b=None, label=None, text_c=None):
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
   
def create_examples(data_df):
    answer = -1
    examples = []
    for num in range(len(data_df)):
        for k in range(4):
            if data_df.loc[num, 'candidate_' + str(k)] == data_df.loc[num, 'answer']:
                answer = str(k)
        label = convert_to_unicode(answer)

        for k in range(4):
            text_a = convert_to_unicode(data_df.loc[num, 'text'])
            text_b = convert_to_unicode(data_df.loc[num, 'candidate_' + str(k)])
            if text_b == 'none':
                text_b = ''
            text_c = convert_to_unicode(data_df.loc[num, 'question'])
            examples.append(
                InputExample(text_a=text_a, text_b=text_b, label=label, text_c=text_c))
    return examples

def create_ori_examples(data_df):
    answer = -1
    examples = []
    for num in range(len(data_df)):
        for k in range(4):
            if data_df.loc[num, 'candidate_' + str(k)] == data_df.loc[num, 'answer']:
                answer = str(k)
        label = convert_to_unicode(answer)

        for k in range(4):
            text_a = convert_to_unicode(data_df.loc[num, 'ori_text'])
            text_b = convert_to_unicode(data_df.loc[num, 'candidate_' + str(k)])
            if text_b == 'none':
                text_b = ''
            text_c = convert_to_unicode(data_df.loc[num, 'question'])
            examples.append(
                InputExample(text_a=text_a, text_b=text_b, label=label, text_c=text_c))
    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):

    """Loads a data file into a list of `InputBatch`s."""
    """ 各样本格式：CLS document SEP question SEP choice SEP
    """

    print("#examples", len(examples))
    labels = ["0", "1", "2", "3"]
    label_map = {}
    for (i, label) in enumerate(labels):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in tqdm(enumerate(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_seq_length - 4:
                break
            if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
                tokens_b.pop()
            else:
                tokens_c.pop()

        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
        if len(features[-1]) == 4:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features


def get_dataset(examples, tokenizer, max_length, is_test=False, use_cache=True):
    """ 每一个样本有4个子样本组成，子样本是各个选项和问题、上下文的拼接，子样本间共享label
            CLS doc SEP question SEP choice-A SEP
            CLS doc SEP question SEP choice-B SEP
            CLS doc SEP question SEP choice-C SEP
            CLS doc SEP question SEP choice-D SEP
        因此，维度是 batch size * 4 * seq length
    """
    features = convert_examples_to_features(examples, max_length, tokenizer)
    input_ids, input_mask, segment_ids = [], [], []
    labels = ["0", "1", "2", "3"]
    label_id = []
    for f in features:
        # new_ids, new_mask, new_segment_ids = [], [], []
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(len(labels)):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)

        if not is_test:
            label_id.append([f[0].label_id])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    if not is_test:
        all_label_ids = torch.tensor(label_id, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


if __name__ == '__main__':
    pass

