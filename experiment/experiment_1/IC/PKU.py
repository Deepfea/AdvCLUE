import numpy as np
import jieba
import pandas as pd
import os

def get_IC(dataset_path):

    train_dataset_path = os.path.join(dataset_path, 'pku_training.npz')
    train_data = np.load(train_dataset_path, allow_pickle=True)
    x_train = train_data["words"]
    text_list = get_data(x_train)
    train_seg = seg_data(text_list)

    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    text_list = get_data(x_test)
    test_seg = seg_data(text_list)


    total_seg = []
    total_seg.extend(train_seg)
    total_seg.extend(test_seg)


    x = len(total_seg)
    y = len(set(total_seg))

    result = float(y) / float(x)
    return result

def get_data(x):
    temp_list = []
    for data_num in range(len(x)):
        temp_str = ''
        for str_num in range(len(x[data_num])):
            temp_str += x[data_num][str_num]
        temp_list.append(temp_str)
    return temp_list

def seg_data(text_list):
    temp_str_0 = ''
    temp_str_1 = ''
    temp_str_2 = ''
    seg_list = []
    for data_num in range(len(text_list)):
        temp_str = text_list[data_num]
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
