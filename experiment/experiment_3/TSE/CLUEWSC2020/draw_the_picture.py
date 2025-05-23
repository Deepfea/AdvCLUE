import os
import pandas as pd
import torch
from transformers import BertTokenizer
from method.step1_adv_sample_generation.CLUEWSC2020.dataset import get_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT
import numpy as np
from tqdm import tqdm
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def get_output(dev_dataloader, model):
    config = Config()
    model.eval()
    test_flag = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(dev_dataloader, 0)):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
            outputs = model(input)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = outputs
                total_label = targets
            else:
                total_npy = np.concatenate((total_npy, outputs), axis=0)
                total_label = np.concatenate((total_label, targets), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(total_npy)
    return output_arr, total_label

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(base_tokenizer_path, model_name))

    config = Config()
    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    dev_dataloader = get_dataloader(adv_data, tokenizer, config.max_length, config.batch_size)

    config.pretrained_model = os.path.join(base_model_path, 'ori_model', dataset_name, model_name)
    model = BERT(config)
    model = model.to(config.device)
    output_arr1, label_list1 = get_output(dev_dataloader, model)

    config.pretrained_model = os.path.join(base_model_path, 'retrained_model', dataset_name, model_name)
    model = BERT(config)
    model = model.to(config.device)
    output_arr2, label_list2 = get_output(dev_dataloader, model)

    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

if __name__ == "__main__":
    pass


