import numpy as np
import jieba
import pandas as pd
import os

def get_ST(dataset_path):

    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    num1 = get_num(train_data)
    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    num2 = get_num(test_data)

    total_num = []
    total_num.extend(num1)
    total_num.extend(num2)

    total_num = np.array(total_num)
    result = np.average(total_num)

    return result


def get_num(data_df):
    text_list_1 = list(data_df['abs'])
    text_list_2 = list(data_df['keyword'])
    num_list = []
    for data_num in range(len(text_list_1)):
       num_list.append(len(text_list_1[data_num]) + len(text_list_2[data_num]))
    return num_list


if __name__ == '__main__':
    pass
