import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.LCQMC.load_token import load_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.LCQMC.bert_model import BertClassifier
from method.step1_adv_sample_generation.LCQMC.Data import *
from sklearn.metrics import accuracy_score

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/LCQMC/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model = BertClassifier(config)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)

    test_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    acc1 = get_acc(model, test_loader)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    acc2 = get_acc(model, test_loader)

    print((acc1 + acc2) / 2.0)



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
