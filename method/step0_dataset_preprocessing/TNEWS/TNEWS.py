import json
import os
import pandas as pd
from flatten_json import flatten

dataset_name = 'TNEWS'
base_path = '/media/usr/external/home/usr/project/project3_data/dataset'
dataset_path = os.path.join(base_path, dataset_name)
train_data_path = os.path.join(dataset_path, 'train.json')
train_data = open(train_data_path, 'r', encoding='UTF-8')
complex_json_data = []
for line in train_data.readlines():
    dic = json.loads(line)
    complex_json_data.append(dic)
flat_json_data = [flatten(item) for item in complex_json_data]
df_complex = pd.DataFrame(flat_json_data)
select_col = ['sentence', 'label']
flat_data = df_complex[select_col]
flat_data.columns = ['text', 'label']
print(flat_data)
for num in range(len(flat_data)):
    temp = flat_data.loc[num, 'label']
    temp = int(temp) % 100
    if temp == 16:
        temp = 5
    elif temp == 15:
        temp = 11
    flat_data.loc[num, 'label'] = temp
print(flat_data)
flat_data.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)

test_data_path = os.path.join(dataset_path, 'dev.json')
test_data = open(test_data_path, 'r', encoding='UTF-8')
complex_json_data = []
for line in test_data.readlines():
    dic = json.loads(line)
    complex_json_data.append(dic)
flat_json_data = [flatten(item) for item in complex_json_data]
df_complex = pd.DataFrame(flat_json_data)
select_col = ['sentence', 'label']
flat_data = df_complex[select_col]
flat_data.columns = ['text', 'label']
print(flat_data)
for num in range(len(flat_data)):
    temp = flat_data.loc[num, 'label']
    temp = int(temp) % 100
    if temp == 16:
        temp = 5
    elif temp == 15:
        temp = 11
    flat_data.loc[num, 'label'] = temp
print(flat_data)
flat_data.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)

