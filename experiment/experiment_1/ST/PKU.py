import numpy as np
import jieba
import pandas as pd
import os

def get_ST(dataset_path):

    train_dataset_path = os.path.join(dataset_path, 'pku_training.npz')
    train_data = np.load(train_dataset_path, allow_pickle=True)
    x_train = train_data["words"]
    num1 = get_num_npz(x_train)


    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    num2 = get_num_npz(x_test)

    total_num = []
    total_num.extend(num1)
    total_num.extend(num2)

    total_num = np.array(total_num)
    result = np.average(total_num)

    return result


def get_num(data_df):
    text_list = list(data_df['text'])
    num_list = []
    for data_num in range(len(text_list)):
       num_list.append(len(text_list[data_num]))
    return num_list

def get_num_npz(x_train):
    num_list = []
    for data_num in range(len(x_train)):
       num_list.append(len(x_train[data_num]))
    return num_list


if __name__ == '__main__':
    pass
