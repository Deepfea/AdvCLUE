import numpy as np
import os
import torch
from experiment.experiment_4.cal_right_token_num.CLUEWSC2020.dataset import get_dataloader
import pandas as pd
from transformers import BertTokenizer


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''


def get_acc(dev_dataloader, model):
    config = Config()
    model.eval()
    results = []
    with torch.no_grad():
        for _, data in enumerate(dev_dataloader, 0):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
            outputs = model(input)
            big_val, big_idx = torch.max(outputs, dim=1)
            results.extend((big_idx == targets).cpu().clone().numpy().tolist())

    return results


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):

    adv_df = load_adv_data(dataset_name, base_adv_path)

    config = Config()
    config.pretrained_model = os.path.join(base_model_path, dataset_name, model_name)


    tokenizer_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    dev_dataloader, data_set = get_dataloader(adv_df, tokenizer, config.max_length, config.batch_size)


    get_token1(data_set)


def get_token1(test_dataset):
    token_list = []
    for num in range(len(test_dataset)):
        temp_list = test_dataset[num]['input_ids'].clone().numpy().tolist()
        token_list.extend(temp_list)
    token_arr = np.unique(np.array(token_list))
    print(len(token_arr)-1)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == "__main__":

    pass