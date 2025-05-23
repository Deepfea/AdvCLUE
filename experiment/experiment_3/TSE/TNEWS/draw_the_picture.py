import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from tqdm import tqdm
from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE

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
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=256)
    model.eval()
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
                # print(len(temp_npy))
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

def evaluate(dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    test_loader = DataLoader(dataset, batch_size=256)
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(dataset): .4f}')

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(base_tokenizer_path, model_name))

    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    print(len(adv_data))
    test_dataset = MyDataset(adv_data, tokenizer)

    load_model_path = os.path.join(base_model_path, 'ori_model', dataset_name, model_name)
    model = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.pt'))
    output = get_output(test_dataset, model)
    # evaluate(test_dataset, model)
    label_list1 = adv_data['label']
    output_arr1 = np.array(output)

    load_model_path = os.path.join(base_model_path, 'retrained_model', dataset_name, model_name)
    model = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.pt'))
    output = get_output(test_dataset, model)
    # evaluate(test_dataset, model)
    label_list2 = adv_data['label']
    output_arr2 = np.array(output)
    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)



if __name__ == '__main__':

    pass

