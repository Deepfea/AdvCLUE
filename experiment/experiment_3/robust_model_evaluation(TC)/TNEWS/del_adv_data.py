import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from tqdm import tqdm
from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset

label_num = 15
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, label_num)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def evaluate(dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
    total_acc_test = 0
    flag_list = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            temp_flag_list = (output.argmax(dim=1) == test_label).cpu().numpy()
            flag_list.extend(temp_flag_list)
            # print(flag_list)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(len(flag_list))
    return flag_list


def del_data(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    pre_model_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)

    load_model_path = os.path.join(base_model_path, dataset_name, model_name)
    model = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_data = load_adv_data(dataset_name, base_adv_path)
    print(len(test_data))
    test_dataset = MyDataset(test_data, tokenizer)
    result = evaluate(test_dataset, model)
    del_num = int(len(result) * 0.5)

    belong_list = []
    ori_text_list = []
    text_list = []
    str_list = []
    word_list = []
    label_list = []
    type_list = []
    error_num_list = []
    temp_num = 0
    for num in range(len(result)):
        if result[num] == False and temp_num < del_num:
            temp_num += 1
            continue

        belong_list.append(test_data.loc[num, 'belong'])
        ori_text_list.append(test_data.loc[num, 'ori_text'])
        str_list.append(test_data.loc[num, 'str'])
        word_list.append(test_data.loc[num, 'word'])
        text_list.append(test_data.loc[num, 'text'])
        label_list.append(test_data.loc[num, 'label'])
        type_list.append(test_data.loc[num, 'type'])
        error_num_list.append(test_data.loc[num, 'error_num'])

    merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list, 'word': word_list,
                     'text': text_list, 'label': label_list, 'type': type_list, 'error_num': error_num_list}

    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))

    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    train_data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    return train_data

if __name__ == '__main__':
    pass

