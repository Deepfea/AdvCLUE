import numpy as np
import pandas as pd
from experiment.experiment_2.ori_model_train.CSL.load_token import load_token
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

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
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
    acc1 = get_acc(model, test_loader)
    print(acc1)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    train_data = pd.read_csv(os.path.join(data_path, 'filtered_mutants.csv'))
    return train_data

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
    avg_accu = accuracy_score(reference_list, prediction_list) *100
    return avg_accu

if __name__ == '__main__':
    pass
