import numpy as np
import pandas as pd
from experiment.experiment_2.ori_model_train.LCQMC.load_token import load_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.LCQMC.bert_model import BertClassifier
from experiment.experiment_2.ori_model_train.LCQMC.Data import *
from sklearn.metrics import accuracy_score

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = ''
    pretrained_model = ''

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    test_df = load_adv_data(dataset_name, base_adv_path)
    test_json = load_token(test_df, config.pretrained_model)

    get_token(test_json['input_ids'])

def get_token(test_dataset):
    token_list = []
    for num in range(len(test_dataset)):
        temp_list = test_dataset[num]
        token_list.extend(temp_list)
    token_arr = np.unique(np.array(token_list))
    print(len(token_arr))

def get_acc(model, dev_loader):
    model.eval()
    prediction_list = []
    reference_list = []
    for idx, batch in enumerate(dev_loader):
        loss, prediction, output = model(batch)
        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].clone().numpy())
        prediction_list.extend(prediction)
        reference_list.extend(labels)
    results = []
    for num in range(len(reference_list)):
        if reference_list[num] == prediction_list[num]:
            results.append(True)
        else:
            results.append(False)
    return results

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return train_data

if __name__ == '__main__':

    pass
