import numpy as np
import pandas as pd
from experiment.experiment_2.ori_model_train.CMRC2018.load_token import get_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from experiment.experiment_2.ori_model_train.CMRC2018.create_dataset import SquadDataset
import torch

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = ''
    pretrained_model = ''

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    load_model_path = os.path.join(base_model_path, dataset_name, model_name)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)

    test_df = load_adv_data(dataset_name, base_adv_path)
    test_encodings = get_token(test_df, config.pretrained_model)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    acc1_1, acc1_2 = get_acc(model, test_loader)

    result = (acc1_1 + acc1_2) / 2.0
    print(1- result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

def get_acc(model, dev_loader):
    model.eval()
    start_list = []
    end_list = []
    for idx, batch in enumerate(dev_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)

        acc_start = (start_pred == start_positions).float().cpu().numpy()
        acc_end = (end_pred == end_positions).float().cpu().numpy()

        start_list += list(acc_start)
        end_list += list(acc_end)
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    acc1 = np.average(start_list)
    acc2 = np.average(end_list)
    return acc1, acc2

if __name__ == '__main__':
    pass
