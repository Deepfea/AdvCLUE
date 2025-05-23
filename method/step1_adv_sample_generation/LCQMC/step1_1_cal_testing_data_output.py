import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.LCQMC.load_token import load_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.LCQMC.bert_model import BertClassifier
from method.step1_adv_sample_generation.LCQMC.Data import *
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model = BertClassifier(config)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint)
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    output_arr = get_output(model, test_loader)
    output_list = []
    for temp_num in range(len(output_arr)):
        temp_npy = output_arr[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(output_list)
    # print(output_arr)
    save_data_path = os.path.join(save_path, 'source_data')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    np.save(os.path.join(save_data_path, 'testing_data_output.npy'), output_arr)

    test_df.to_csv(os.path.join(save_data_path, 'test.csv'), index=False)

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
    avg_accu = accuracy_score(reference_list, prediction_list) * 100
    print(avg_accu)
    return np.array(output_list)

if __name__ == '__main__':
    pass
