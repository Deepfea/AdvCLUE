import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from method.step1_adv_sample_generation.TNEWS.seg_data_to_words import seg_data
from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset
from tqdm import tqdm

label_num = 15

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, len(label_num))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)

def gen_new_sentence(data_path):
    belong_list = []
    str_list = []
    fact_list = []
    label_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)
    for data_num in range(len(seg_arr)):
        temp_segs = list(set(seg_arr[data_num]))
        ori_fact = test_data.loc[data_num, 'text']
        label = test_data.loc[data_num, 'label']
        for seg_num in range(len(temp_segs)):
            temp_seg = temp_segs[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            temp_fact = ori_fact.replace(temp_seg, '')
            belong_list.append(data_num)
            str_list.append(temp_seg)
            fact_list.append(temp_fact)
            label_list.append(label)
    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'text': fact_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    # data_df.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)
    return data_df

def cal_word_importance(save_path, base_model_path):
    data_path = os.path.join(save_path, 'source_data')
    segmentation(data_path)
    new_data_df = gen_new_sentence(data_path)
    tokenizer = BertTokenizer.from_pretrained(base_model_path)

    test_dataset = MyDataset(new_data_df, tokenizer)

    model_load_path = os.path.join(save_path, 'bert_base_chinese', 'best_TNEWS_bert_base_chinese.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            temp_len = len(test_label)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if temp_len == 1:
                temp_list = []
                temp_list.append(temp_npy)
                temp_npy = np.array(temp_list)
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)

    source_data_output = np.load(os.path.join(data_path, 'testing_data_output.npy'))
    importance_value = []
    for data_num in range(len(output_list)):
        temp_output = np.array(output_list[data_num])
        temp_label = new_data_df.loc[data_num, 'label']
        temp_value = source_data_output[new_data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    new_data_df['importance_value'] = importance_value
    print(len(new_data_df))
    df_filtered = new_data_df[new_data_df['importance_value'] > -100].reset_index(drop=True)
    print(len(new_data_df))
    df_filtered.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)
    seg_list = []
    seg_score = []
    for class_num in range(len(source_data_output)):
        seg_list.append([])
        seg_score.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
    # print(seg_list)
    # print(seg_score)
    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)
    np.save(os.path.join(data_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(data_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))


if __name__ == '__main__':
    pass
