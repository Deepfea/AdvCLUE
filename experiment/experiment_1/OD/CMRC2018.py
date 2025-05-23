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
    starts1, ends1 = get_output(model, test_loader)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_encodings = get_token(test_df)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    starts2, ends2 = get_output(model, test_loader)

    starts = starts1 + starts2

    ends = ends1 + ends2

    np.save(os.path.join(save_path, 'start_output.npy'), starts)
    np.save(os.path.join(save_path, 'end_output.npy'), ends)

    result1 = get_gini(starts)
    result2 = get_gini(ends)

    print(result1)
    print(result2)
    print((result1+result2) / 2.0)


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
    acc_start_sum = 0.0
    acc_end_sum = 0.0
    model.eval()
    start_list = []
    end_list = []
    for idx, batch in enumerate(tqdm(dev_loader)):
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
        temp_start = outputs.start_logits.detach().cpu().clone().numpy()
        start_list.extend(temp_start)
        temp_end = outputs.end_logits.detach().cpu().clone().numpy()
        end_list.extend(temp_end)

        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)

        acc_start = (start_pred == start_positions).float().mean()
        acc_end = (end_pred == end_positions).float().mean()

        acc_start_sum += acc_start
        acc_end_sum += acc_end


    start_arr = []
    for temp_num in range(len(start_list)):
        temp_npy = start_list[temp_num]
        temp_npy = softmax(temp_npy)
        start_arr.append(temp_npy)

    end_arr = []
    for temp_num in range(len(end_list)):
        temp_npy = end_list[temp_num]
        temp_npy = softmax(temp_npy)
        end_arr.append(temp_npy)

    return start_arr, end_arr



if __name__ == '__main__':
    pass
