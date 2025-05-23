import numpy as np
import pandas as pd
from method.step2_calculate_metric.CMRC2018.load_token import get_token,get_ori_token
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
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    load_model_path = os.path.join(base_model_path, dataset_name, model_name)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)

    test_df1 = load_adv_data(dataset_name, base_adv_path)

    context_list = []
    question_list = []
    answer_start_list = []
    answer_end_list = []
    for data_num in range(len(test_df1)):
        context_list.append(test_df1.loc[data_num, 'ori_context'])
        question_list.append(test_df1.loc[data_num, 'ori_question'])
        answer_start_list.append(test_df1.loc[data_num, 'ori_answer_start'])
        answer_end_list.append(test_df1.loc[data_num, 'ori_answer_end'])
    merge_dt_dict = {'context': context_list, 'question': question_list, 'answer_start': answer_start_list,
                     'answer_end': answer_end_list}
    test_df2 = pd.DataFrame(merge_dt_dict)
    test_df = pd.concat([test_df1, test_df2], axis=0)

    test_encodings = get_token(test_df, config.pretrained_model)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    outputs1, outputs2 = get_acc(model, test_loader)
    outputs1_1 = outputs1[:len(outputs1) // 2]
    outputs2_1 = outputs1[len(outputs1) // 2:]
    outputs1_2 = outputs2[:len(outputs2) // 2]
    outputs2_2 = outputs2[len(outputs2) // 2:]

    result1 = get_entropy(outputs1_1, outputs2_1)
    result2 = get_entropy(outputs1_2, outputs2_2)

    print((result1+result2) / 2.0)


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
        start_list.extend(outputs.start_logits.cpu().clone().detach().numpy().tolist())
        end_list.extend(outputs.end_logits.cpu().clone().detach().numpy().tolist())
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    print(start_list.shape)
    print(end_list.shape)
    return start_list, end_list


if __name__ == '__main__':
    pass
