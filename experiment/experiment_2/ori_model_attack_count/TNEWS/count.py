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
    results = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            results.extend((output.argmax(dim=1) == test_label).cpu().clone().numpy().tolist())

    print(f'Test Accuracy: {total_acc_test / len(dataset): .4f}')
    return total_acc_test / float(len(dataset)), results


def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    pre_model_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)

    load_model_path = os.path.join(base_model_path, dataset_name, model_name)
    model = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_data = load_adv_data(dataset_name, base_adv_path)
    # print(len(test_data))
    test_dataset = MyDataset(test_data, tokenizer)
    acc, results = evaluate(test_dataset, model)
    # print(results)
    error_num_list, total_num_list = get_num(test_data, results)
    print(error_num_list)
    print(total_num_list)


def get_num(df_test, results):
    mut_name_list = ['word_shuffling', 'character_deleting',
                     'symbol_insertion', 'glyph_replacement',
                     'character_splitting', 'homophone_replacement',
                     'nasal_replacement', 'dorsal_replacement',
                     'context_prediction', 'synonym_replacement',
                     'traditional_conversion']
    error_num_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_num_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for num in range(len(df_test)):
        type_name = df_test.loc[num, 'type']
        temp_index = mut_name_list.index(type_name)
        total_num_list[temp_index] += 1
        if results[num] == 0 or results[num] == False:
            error_num_list[temp_index] += 1
    return error_num_list, total_num_list

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == '__main__':

    pass

