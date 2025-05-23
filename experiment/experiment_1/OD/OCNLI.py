import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.OCNLI.load_token import load_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.OCNLI.bert_model import BertClassifier
from method.step1_adv_sample_generation.OCNLI.Data import *

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/OCNLI'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/OCNLI/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model = BertClassifier(config)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)

    test_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    output1 = get_output(model, test_loader)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    output2 = get_output(model, test_loader)

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

def get_output(model, dev_loader):
    model.eval()
    output_list = []
    prediction_list = []
    reference_list = []
    for idx, batch in tqdm(enumerate(dev_loader)):
        loss, prediction, output = model(batch)
        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].clone().numpy())
        prediction_list.extend(prediction)
        reference_list.extend(labels)
        outputs = output.cpu().clone().numpy()
        output_list.extend(outputs)
    output_arr = []
    for temp_num in range(len(output_list)):
        temp_npy = output_list[temp_num]
        temp_npy = softmax(temp_npy)
        output_arr.append(temp_npy)
    output_arr = np.array(output_arr)
    return output_arr

if __name__ == '__main__':
    pass
