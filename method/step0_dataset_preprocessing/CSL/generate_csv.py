import os

import pandas as pd
import json

def get_wsc_json(file):
    with open(file, "r") as f:
        datas = []
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas

def generate_df(data_df):
    abs_list = []
    keyword_list = []
    label_list = []
    x1_list = data_df['abst']
    x2_list = data_df['keyword']
    y_list = data_df['label']
    print(len(data_df))
    for num in range(len(x1_list)):
        if len(x1_list[num]) + len(x2_list[num]) > 450:
            continue
        abs_list.append(x1_list[num])
        temp_keyword = ''
        for keyword_num in range(len(x2_list[num])):
            if keyword_num != 0:
                temp_keyword += ';'
            temp_keyword += x2_list[num][keyword_num]
        temp_keyword += ';'
        keyword_list.append(temp_keyword)
        label_list.append(y_list[num])
    merge_dt_dict = {'abs': abs_list, 'keyword': keyword_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))
    return data_df


def save_csv(dataset_path):
    print(os.path.join(dataset_path, 'train.csv'))
    datas = get_wsc_json("/media/usr/external/home/usr/project/project3_data/dataset/CSL/train.json")
    temp_df = pd.DataFrame(datas)
    train_df = generate_df(temp_df)
    train_df.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)

    print(os.path.join(dataset_path, 'test.csv'))
    datas = get_wsc_json("/media/usr/external/home/usr/project/project3_data/dataset/CSL/dev.json")
    temp_df = pd.DataFrame(datas)
    dev_df = generate_df(temp_df)
    dev_df.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)

if __name__ == "__main__":
    pass

