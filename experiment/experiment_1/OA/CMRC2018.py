import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.CMRC2018.load_token import get_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset
import torch

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/CMRC2018/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)

    test_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_encodings = get_token(test_df)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    acc1_1, acc1_2 = get_acc(model, test_loader)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_encodings = get_token(test_df)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    acc2_1, acc2_2 = get_acc(model, test_loader)

    result1 = (acc1_1 + acc2_1) / 2.0
    result2 = (acc1_2 + acc2_2) / 2.0
    result = (result1 + result2) / 2.0
    print((acc1_1 + acc2_1) / 2.0)
    print((acc1_2 + acc2_2) / 2.0)
    print(result)



def get_acc(model, dev_loader):
    model.eval()
    acc_start_sum = 0.0
    acc_end_sum = 0.0
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
