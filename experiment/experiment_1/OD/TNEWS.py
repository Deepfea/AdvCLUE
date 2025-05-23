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

def get_output(test_dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(output_list)
    return output_arr

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    model = torch.load(os.path.join(base_model_path, 'best_TNEWS_bert_base_chinese.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_dataset = MyDataset(train_data, tokenizer)
    output1 = get_output(train_dataset, model)

    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_dataset = MyDataset(test_data, tokenizer)
    output2 = get_output(test_dataset, model)

    output = np.concatenate((output1, output2), axis=0)
    output_arr = np.array(output)

    np.save(os.path.join(save_path, 'output.npy'), output_arr)

    result = get_gini(output_arr)
    print(result)

def get_gini(output_list):
    value_list = []
    for output_num in range(len(output_list)):
        value = 0
        temp_output = output_list[output_num]
        for num in range(len(temp_output)):
            value = value + temp_output[num] * temp_output[num]
        value_list.append(value)
    value_arr = np.array(value_list)
    result = np.std(value_arr)
    return result

if __name__ == '__main__':
    pass
