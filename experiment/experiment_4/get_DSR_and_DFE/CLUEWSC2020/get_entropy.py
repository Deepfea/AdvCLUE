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


def evaluate_data(dev_dataloader, model):
    outputs = []
    config = Config()
    model.eval()
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
            output = model(input).cpu().clone().numpy().tolist()

            outputs.extend(output)
    outputs = np.array(outputs)
    # print(outputs)

    return outputs

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def get_entropy(outputs1, outputs2):
    results = []
    for num in range(len(outputs1)):
        temp_output1 = outputs1[num]
        temp_output2 = outputs2[num]
        dot_product = np.dot(temp_output1, temp_output2)
        norm_temp_output1 = np.linalg.norm(temp_output1)
        norm_temp_output2 = np.linalg.norm(temp_output2)
        similarity = dot_product / (norm_temp_output1 * norm_temp_output2)
        results.append(similarity)
    results = np.array(results)
    total_sum = np.sum(results)
    final_result = 0
    for result_num in range(len(results)):
        temp_result = - results[result_num] / total_sum * np.log2(results[result_num] / total_sum)
        if np.isnan(temp_result):
            temp_result = 1
        final_result += temp_result
    return final_result


def cal_entropy(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):

    adv_df = load_adv_data(dataset_name, base_adv_path)

    config = Config()
    config.pretrained_model = os.path.join(base_model_path, dataset_name, model_name)
    model = BERT(config)
    model = model.to(config.device)

    tokenizer_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    dev_dataloader = get_dataloader(adv_df, tokenizer, config.max_length, config.batch_size)
    outputs1 = evaluate_data(dev_dataloader, model)

    dataset_path = '/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020/test.csv'
    test_data = pd.read_csv(dataset_path)

    text_list = []
    span1_begin_list = []
    span1_end_list = []
    span2_begin_list = []
    span2_end_list = []
    label_id_list = []
    for data_num in range(len(adv_df)):

        temp_num = adv_df.loc[data_num, 'belong']
        text_list.append(test_data.loc[temp_num, 'text'])
        span1_begin_list.append(test_data.loc[temp_num, 'span1_begin'])
        span1_end_list.append(test_data.loc[temp_num, 'span1_end'])
        span2_begin_list.append(test_data.loc[temp_num, 'span2_begin'])
        span2_end_list.append(test_data.loc[temp_num, 'span2_end'])
        label_id_list.append(test_data.loc[temp_num, 'label_id'])

    merge_dt_dict = {'text': text_list, 'span1_begin': span1_begin_list, 'span1_end': span1_end_list,
                     'span2_begin': span2_begin_list, 'span2_end': span2_end_list, 'label_id': label_id_list}
    data_df = pd.DataFrame(merge_dt_dict)
    dev_dataloader = get_dataloader(data_df, tokenizer, config.max_length, config.batch_size)
    outputs2 = evaluate_data(dev_dataloader, model)

    result = get_entropy(outputs1, outputs2)

    print(result)


def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == "__main__":
    pass




