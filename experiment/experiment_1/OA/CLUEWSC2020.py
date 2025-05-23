import numpy as np
import os
import torch
from method.step1_adv_sample_generation.CLUEWSC2020.dataset import get_dev_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT
from tqdm import tqdm

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
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    return epoch_accu


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def evaluate_testing_data(dataset_path, base_model_path, save_path):
    config = Config()
    config.pretrained_model = base_model_path
    model = BERT(config)
    model = model.to(config.device)

    dataset_path_path = os.path.join(dataset_path, 'train.csv')
    dev_dataloader, def_df = get_dev_dataloader(dataset_path_path, config.max_length, config.batch_size)
    acc1 = get_acc(dev_dataloader, model)

    dataset_path_path = os.path.join(dataset_path, 'test.csv')
    dev_dataloader, def_df = get_dev_dataloader(dataset_path_path, config.max_length, config.batch_size)
    acc2 = get_acc(dev_dataloader, model)

    print((acc1 + acc2) / 2.0)


if __name__ == "__main__":
    pass
