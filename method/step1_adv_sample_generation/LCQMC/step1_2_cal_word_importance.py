import numpy as np
import os
import pandas as pd
from method.step1_adv_sample_generation.LCQMC.seg_data_to_words import seg_data
from method.step1_adv_sample_generation.LCQMC.bert_model import BertClassifier
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.LCQMC.load_token import load_token
from method.step1_adv_sample_generation.LCQMC.Data import *
from tqdm import tqdm

def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def gen_new_sentence(data_path):
    belong_list = []
    belong_text = []
    str_list = []
    fact_a_list = []
    fact_b_list = []
    label_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)
    for data_num in range(len(seg_arr)):
        temp_segs = seg_arr[data_num]
        ori_fact_a = test_data.loc[data_num, 'text_a']
        ori_fact_b = test_data.loc[data_num, 'text_b']
        label = test_data.loc[data_num, 'label']
        segs_a = temp_segs[0]
        segs_b = temp_segs[1]
        for seg_num in range(len(segs_a)):
            temp_seg = segs_a[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            temp_fact = ori_fact_a.replace(temp_seg, '')
            belong_list.append(data_num)
            belong_text.append('a')
            str_list.append(temp_seg)
            fact_a_list.append(temp_fact)
            fact_b_list.append(ori_fact_b)
            label_list.append(label)
        for seg_num in range(len(segs_b)):
            temp_seg = segs_b[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            temp_fact = ori_fact_b.replace(temp_seg, '')
            belong_list.append(data_num)
            belong_text.append('b')
            str_list.append(temp_seg)
            fact_a_list.append(ori_fact_a)
            fact_b_list.append(temp_fact)
            label_list.append(label)

    merge_dt_dict = {'belong': belong_list, 'belong_text': belong_text, 'str': str_list, 'text_a': fact_a_list,
                     'text_b': fact_b_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    print(data_df)
    return data_df

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 256
    batch_size = 32
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def cal_word_importance(save_path, base_model_path):
    config = Config()

    data_path = os.path.join(save_path, 'source_data')
    segmentation(data_path)
    new_data_df = gen_new_sentence(data_path)
    test_json = load_token(new_data_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)

    model = BertClassifier(config)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)

    model.eval()
    temp_list = []
    for idx, batch in tqdm(enumerate(test_loader)):
        loss, prediction, output = model(batch)
        outputs = output.cpu().clone().numpy()
        temp_list.extend(outputs)
    output_list = []
    for temp_num in range(len(temp_list)):
        temp_npy = temp_list[temp_num]
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
    seg_text = []
    for class_num in range(len(source_data_output)):
        seg_list.append([])
        seg_score.append([])
        seg_text.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
        seg_text[class_name].append(df_filtered.loc[df_filtered_num, 'belong_text'])
    print(seg_list)
    print(seg_score)
    print(seg_text)

    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)
    seg_text_arr = np.array(seg_text)

    np.save(os.path.join(data_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(data_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))
    np.save(os.path.join(data_path, 'seg_text.npy'), seg_text_arr)
    print(len(seg_text_arr))


if __name__ == '__main__':
    pass
