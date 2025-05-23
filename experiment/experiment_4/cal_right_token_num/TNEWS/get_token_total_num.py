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
    results = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            results.extend((output.argmax(dim=1) == test_label).cpu().clone().numpy().tolist())
            # print(results)
    return results


def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    pre_model_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)


    test_data = load_adv_data(dataset_name, base_adv_path)
    test_dataset = MyDataset(test_data, tokenizer)

    get_token_total_num(test_dataset)


def get_token_total_num(test_dataset):
    token_list = []
    for num in range(len(test_dataset.texts)):
        temp_list = test_dataset.texts[num]['input_ids'].clone().numpy()[0].tolist()
        token_list.extend(temp_list)
    token_arr = np.unique(np.array(token_list))
    print(len(token_arr))

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == '__main__':

    pass
