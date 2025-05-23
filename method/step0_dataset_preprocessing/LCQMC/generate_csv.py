import json
import os
import numpy as np
import pickle

import pandas as pd
from transformers import BertTokenizer


def truncate_seq_pair(str_a, str_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    len_a = len(str_a)
    len_b = len(str_b)
    while True:
        total_length = len_a + len_b
        if total_length <= max_length:
            break
        if len_a > len_b:
            len_a -= 1
        else:
            len_b -= 1
    fina_str_a = str_a[:len_a]
    fina_str_b = str_b[:len_b]
    return fina_str_a, fina_str_b

def process_set(dataset_path, file_name, max_seq_length):
    tsv_path = os.path.join(dataset_path, file_name + '.tsv')
    text_a_list = []
    text_b_list = []
    label_list = []
    flag = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if flag == 0:
                flag = 1
                continue
            line_list = line.strip().split('\t')
            if(len(line_list)!=3):
                print('data process wrong..')
                exit()
            text_a = line_list[0]
            text_b = line_list[1]
            text_a, text_b = truncate_seq_pair(text_a, text_b, max_seq_length - 5)
            text_a_list.append(text_a)
            text_b_list.append(text_b)
            label_list.append(int(line_list[2]))
    merge_dt_dict = {'text_a': text_a_list, 'text_b': text_b_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(dataset_path, file_name + '.csv'), index=False)

def generate_csv(dataset_path, max_seq_length):
    file_name = 'train'
    process_set(dataset_path, file_name, max_seq_length)

    file_name = 'test'
    process_set(dataset_path, file_name, max_seq_length)

if __name__ == "__main__":
    pass





