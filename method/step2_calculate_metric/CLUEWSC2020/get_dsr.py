import numpy as np
import os
import torch
from experiment.experiment_2.ori_model_train.CLUEWSC2020.dataset import get_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT
from tqdm import tqdm
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
    n_correct = 0
    nb_tr_examples = 0
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
            n_correct += calcuate_accu(big_idx, targets)
            nb_tr_examples += targets.size(0)
    epoch_accu = (n_correct) / nb_tr_examples
    # print(f"Validation Accuracy Epoch: {epoch_accu}")
    return epoch_accu


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):

    adv_df = load_adv_data(dataset_name, base_adv_path)

    config = Config()
    config.pretrained_model = os.path.join(base_model_path, dataset_name, model_name)
    model = BERT(config)
    model = model.to(config.device)

    tokenizer_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    dev_dataloader = get_dataloader(adv_df, tokenizer, config.max_length, config.batch_size)

    acc1 = get_acc(dev_dataloader, model)

    print(1 - acc1)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == "__main__":
    pass
