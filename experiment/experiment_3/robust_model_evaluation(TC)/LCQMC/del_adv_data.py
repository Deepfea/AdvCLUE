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

def del_data(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    load_model_path = os.path.join(base_model_path, dataset_name, model_name)
    model = BertClassifier(config)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    test_df = load_adv_data(dataset_name, base_adv_path)
    print(len(test_df))
    test_json = load_token(test_df, config.pretrained_model)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    flag_list = get_acc(model, test_loader)
    del_num = int(len(flag_list) * 0.5)

    belong_list = []
    ori_text_a_list = []
    ori_text_b_list = []
    text_a_list = []
    text_b_list = []
    str_list = []
    word_list = []
    label_list = []
    type_list = []
    error_num_list = []
    temp_num = 0

    for num in range(len(flag_list)):
        if flag_list[num] == 0 and temp_num < del_num:
            temp_num += 1
            continue
        belong_list.append(test_df.loc[num, 'belong'])
        ori_text_a_list.append(test_df.loc[num, 'ori_text_a'])
        ori_text_b_list.append(test_df.loc[num, 'ori_text_b'])
        str_list.append(test_df.loc[num, 'str'])
        word_list.append(test_df.loc[num, 'word'])
        text_a_list.append(test_df.loc[num, 'text_a'])
        text_b_list.append(test_df.loc[num, 'text_b'])
        label_list.append(test_df.loc[num, 'label'])
        type_list.append(test_df.loc[num, 'type'])
        error_num_list.append(test_df.loc[num, 'error_num'])
    merge_dt_dict = {'belong': belong_list, 'ori_text_a': ori_text_a_list, 'ori_text_b': ori_text_b_list,
                     'str': str_list, 'word': word_list,
                     'text_a': text_a_list, 'text_b': text_b_list, 'label': label_list, 'type': type_list,
                     'error_num': error_num_list}
    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))
    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)

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
    flag_list = []
    for num in range(len(prediction_list)):
        if prediction_list[num] == reference_list[num][0]:
            flag_list.append(True)
        else:
            flag_list.append(False)
    return flag_list

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    train_data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    return train_data

if __name__ == '__main__':
    pass
