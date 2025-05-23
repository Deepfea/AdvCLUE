import numpy as np
import jieba
import pandas as pd
import os
from tqdm import tqdm

def get_IC(dataset_path):

    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_seg = seg_data(train_data)
    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_seg = seg_data(test_data)

    total_seg = []
    total_seg.extend(train_seg)
    total_seg.extend(test_seg)

    x = len(total_seg)
    y = len(set(total_seg))

    result = float(y) / float(x)
    return result


def seg_data(data_df):
    text_list = list(data_df['text'])
    temp_str_0 = ''
    temp_str_1 = ''
    temp_str_2 = ''
    seg_list = []
    for data_num in tqdm(range(len(text_list))):
        temp_str = text_list[data_num]
        print(temp_str)
        temp_str = temp_str.replace('\n', '')
        print(temp_str)
        if temp_str_0 == temp_str[0] and temp_str_1 == temp_str[1] and temp_str_2 == temp_str[2]:
            temp_str_0 = temp_str[0]
            if len(temp_str) > 1:
                temp_str_1 = temp_str[1]
            if len(temp_str) > 2:
                temp_str_2 = temp_str[2]
            continue
        else:
            cut_result = jieba.cut(temp_str, cut_all=False)
            cut_result = list(cut_result)
            seg_list.extend(cut_result)
            temp_str_0 = temp_str[0]
            if len(temp_str) > 1:
                temp_str_1 = temp_str[1]
            if len(temp_str) > 2:
                temp_str_2 = temp_str[2]
    return seg_list


if __name__ == '__main__':
    pass