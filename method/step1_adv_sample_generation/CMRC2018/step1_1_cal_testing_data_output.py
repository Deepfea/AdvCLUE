import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.CMRC2018.load_token import get_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset
import torch

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_encodings = get_token(test_df)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    starts, ends = get_output(model, test_loader)

    start_list = []
    for temp_num in range(len(starts)):
        temp_npy = starts[temp_num]
        temp_npy = softmax(temp_npy)
        start_list.append(temp_npy)
    start_arr = np.array(start_list)

    end_list = []
    for temp_num in range(len(ends)):
        temp_npy = ends[temp_num]
        temp_npy = softmax(temp_npy)
        end_list.append(temp_npy)
    end_arr = np.array(end_list)

    print(start_arr.shape)
    print(end_arr.shape)

    save_data_path = os.path.join(save_path, 'source_data')

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    np.save(os.path.join(save_data_path, 'test_data_start_output.npy'), start_arr)
    np.save(os.path.join(save_data_path, 'test_data_end_output.npy'), end_arr)

    test_df.to_csv(os.path.join(save_data_path, 'test.csv'), index=False)

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
        # print((start_pred == start_positions))
        # print((start_pred == start_positions).float().mean())
        acc_start = (start_pred == start_positions).float().mean()
        acc_end = (end_pred == end_positions).float().mean()

        acc_start_sum += acc_start
        acc_end_sum += acc_end
    print(acc_start_sum / len(dev_loader))
    print(acc_end_sum / len(dev_loader))

    return start_list, end_list

if __name__ == '__main__':
    pass
