import numpy as np
import pandas as pd
from method.step2_calculate_metric.CSL.load_token import load_token, load_ori_token
import os
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.CSL.bert_model import BertClassifier
from experiment.experiment_2.ori_model_train.CSL.Data import *
from sklearn.metrics import accuracy_score

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

    model = BertClassifier(config)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)

    test_df = load_adv_data(dataset_name, base_adv_path)
    test_json = load_token(test_df, config.pretrained_model)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    outputs1 = get_acc(model, test_loader)

    test_df = load_adv_data(dataset_name, base_adv_path)
    test_json = load_ori_token(test_df, config.pretrained_model)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    outputs2 = get_acc(model, test_loader)

    result = get_entropy(outputs1, outputs2)
    print(result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

def get_acc(model, dev_loader):
    model.eval()
    outputs = []
    for idx, batch in enumerate(dev_loader):
        loss, prediction, output = model(batch)
        output = output.cpu().clone().numpy().tolist()
        outputs.extend(output)
    return outputs

if __name__ == '__main__':
    pass
