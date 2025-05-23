import json
import os
import numpy as np
import pickle

import pandas as pd
from transformers import BertTokenizer

def load_token(data):
    vocab_path = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    content = {
        "tokens": list(),
        "input_ids": list(),
        "token_type_ids": list(),
        "mask_ids": list(),
        "labels": list()
    }
    for num in range(len(data)):
        tokens = []
        token_type_ids = []
        text_a = data.loc[num, 'abs']
        text_b = data.loc[num, 'keyword']
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)
        tokens.append("[SEP]")
        token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        content['tokens'].append(tokens)
        content['input_ids'].append(input_ids)
        content['token_type_ids'].append(token_type_ids)
        content['mask_ids'].append(input_mask)
        content['labels'].append(data.loc[num, 'label'])
    assert len(content['tokens']) == len(content['input_ids']) == len(content['token_type_ids']) == len(
        content['mask_ids']) == len(content['labels'])
    return content



if __name__ == "__main__":
    pass



