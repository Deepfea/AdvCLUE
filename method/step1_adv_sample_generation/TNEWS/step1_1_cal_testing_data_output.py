import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn

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


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    tokenizer = BertTokenizer.from_pretrained(base_model_path)

    print("加载数据集......")
    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_dataset = MyDataset(test_data, tokenizer)
    print("加载数据集完成")

    model = torch.load(os.path.join(save_path, 'bert_base_chinese', 'best_TNEWS_bert_base_chinese.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
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
    # print(output_arr)
    save_data_path = os.path.join(save_path, 'source_data')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    np.save(os.path.join(save_data_path, 'testing_data_output.npy'), output_arr)

    test_data.to_csv(os.path.join(save_data_path, 'test.csv'), index=False)

if __name__ == '__main__':
    pass