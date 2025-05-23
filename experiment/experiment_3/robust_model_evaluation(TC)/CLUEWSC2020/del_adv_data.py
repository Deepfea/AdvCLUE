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

    flag_list = []
    with torch.no_grad():
        for _, data in enumerate(dev_dataloader, 0):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end)
            outputs = model(input)
            big_val, big_idx = torch.max(outputs, dim=1)
            result = calcuate_accu(big_idx, targets)
            flag_list.extend(result)
    print(flag_list)
    return flag_list


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    result = (big_idx == targets).cpu().numpy()
    result = list(result)
    return result

def del_data(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):

    adv_df = load_adv_data(dataset_name, base_adv_path)

    config = Config()
    config.pretrained_model = os.path.join(base_model_path, dataset_name, model_name)
    model = BERT(config)
    model = model.to(config.device)

    tokenizer_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    dev_dataloader = get_dataloader(adv_df, tokenizer, config.max_length, config.batch_size)

    flag_list = get_acc(dev_dataloader, model)

    del_num = int(len(flag_list) * 0.5)

    belong_list = []
    ori_text_list = []
    str_list = []
    word_list = []
    text_list = []
    span1_begin_list = []
    span1_end_list = []
    span2_begin_list = []
    span2_end_list = []
    label_id_list = []
    type_list = []
    error_num_list = []
    temp_num = 0
    for num in range(len(flag_list)):
        if flag_list[num] == 0 and temp_num < del_num:
            temp_num += 1
            continue
        belong_list.append(adv_df.loc[num, 'belong'])
        ori_text_list.append(adv_df.loc[num, 'ori_text'])
        str_list.append(adv_df.loc[num, 'str'])
        word_list.append(adv_df.loc[num, 'word'])
        text_list.append(adv_df.loc[num, 'text'])
        span1_begin_list.append(adv_df.loc[num, 'span1_begin'])
        span1_end_list.append(adv_df.loc[num, 'span1_end'])
        span2_begin_list.append(adv_df.loc[num, 'span2_begin'])
        span2_end_list.append(adv_df.loc[num, 'span2_end'])
        label_id_list.append(adv_df.loc[num, 'label_id'])
        type_list.append(adv_df.loc[num, 'type'])
        error_num_list.append(adv_df.loc[num, 'error_num'])
    merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list, 'word': word_list,
                     'text': text_list, 'span1_begin': span1_begin_list, 'span1_end': span1_end_list,
                     'span2_begin': span2_begin_list,
                     'span2_end': span2_end_list, 'label_id': label_id_list, 'type': type_list,
                     'error_num': error_num_list}
    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))
    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    train_data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    return train_data

if __name__ == "__main__":
    pass
