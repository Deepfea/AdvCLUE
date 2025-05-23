import os

import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from tqdm import tqdm
from method.step2_calculate_metric.TNEWS.create_dataset import textDataset, ori_textDataset

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

    def get(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        return pooled_output

def evaluate(dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
    outputs = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model.get(input_id, mask)
            output = output.cpu().clone().numpy().tolist()
            # print(len(output))
            outputs.extend(output)
    return outputs

def get_entropy(outputs1, outputs2):
    results = []
    for num in range(len(outputs1)):
        temp_output1 = outputs1[num]
        temp_output2 = outputs2[num]
        dot_product = np.dot(temp_output1, temp_output2)
        norm_temp_output1 = np.linalg.norm(temp_output1)
        norm_temp_output2 = np.linalg.norm(temp_output2)
        similarity = dot_product / (norm_temp_output1 * norm_temp_output2)
        results.append(similarity)
    results = np.array(results)
    total_sum = np.sum(results)
    final_result = 0
    for result_num in range(len(results)):
        temp_result = - results[result_num] / total_sum * np.log2(results[result_num] / total_sum)
        if np.isnan(temp_result):
            temp_result = 1
        final_result += temp_result
    return final_result

def cal_entropy(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    pre_model_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)
    load_model_path = os.path.join(base_model_path, dataset_name, model_name)
    model = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_data = load_adv_data(dataset_name, base_adv_path)
    # print(len(test_data))
    test_dataset = textDataset(test_data, tokenizer)
    outputs1 = evaluate(test_dataset, model)

    test_dataset = ori_textDataset(test_data, tokenizer)
    outputs2 = evaluate(test_dataset, model)

    result = get_entropy(outputs1, outputs2)
    print(result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == '__main__':
    pass
