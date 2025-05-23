import os
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE
from experiment.experiment_2.ori_model_train.OCNLI.bert_model import BertClassifier
from experiment.experiment_2.ori_model_train.OCNLI.Data import *
from experiment.experiment_2.ori_model_train.OCNLI.load_token import load_token

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = ''
    pretrained_model = ''


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def get_output(model, dev_loader):
    model.eval()
    output_list = []
    prediction_list = []
    reference_list = []
    for idx, batch in tqdm(enumerate(dev_loader)):
        loss, prediction, output = model(batch)
        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].clone().numpy())
        prediction_list.extend(prediction)
        reference_list.extend(labels)
        outputs = output.cpu().clone().numpy()
        output_list.extend(outputs)
    output_arr = []
    label_list = []
    for temp_num in range(len(output_list)):
        label_list.append(reference_list[temp_num][0])
        temp_npy = output_list[temp_num]
        temp_npy = softmax(temp_npy)
        output_arr.append(temp_npy)
    output_arr = np.array(output_arr)

    return output_arr, label_list

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)

    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    test_json = load_token(adv_data, os.path.join(base_tokenizer_path, model_name))
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)

    model = BertClassifier(config)
    checkpoint = torch.load(os.path.join(base_model_path, 'ori_model', dataset_name, model_name, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    output_arr1, label_list1 = get_output(model, test_loader)

    model = BertClassifier(config)
    checkpoint = torch.load(os.path.join(base_model_path, 'retrained_model', dataset_name, model_name, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    output_arr2, label_list2 = get_output(model, test_loader)

    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

if __name__ == "__main__":
    pass


