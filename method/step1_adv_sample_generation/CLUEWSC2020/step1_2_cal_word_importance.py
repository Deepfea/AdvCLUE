import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from method.step1_adv_sample_generation.CLUEWSC2020.seg_data_to_words import seg_data
from method.step1_adv_sample_generation.CLUEWSC2020.dataset import get_temp_data
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr, poi_arr, words_num_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)
    np.save(os.path.join(data_path, 'poi_arr.npy'), poi_arr)
    np.save(os.path.join(data_path, 'words_num_arr.npy'), words_num_arr)

def get_new_data(ori_fact, temp_seg, temp_poi, data_span1_begin, data_span1_end, data_span2_begin, data_span2_end):
    fact = ''
    num = 0
    while num < len(ori_fact):
        if num < int(temp_poi[0]) or num > int(temp_poi[1]):
            fact += ori_fact[num]
        num += 1
    if temp_poi[1] < data_span1_begin:
        temp_data_span1_begin = data_span1_begin - len(temp_seg)
        temp_data_span1_end = data_span1_end -len(temp_seg)
        temp_data_span2_begin = data_span2_begin - len(temp_seg)
        temp_data_span2_end = data_span2_end - len(temp_seg)
    elif temp_poi[0] >= data_span2_end:
        temp_data_span1_begin = data_span1_begin
        temp_data_span1_end = data_span1_end
        temp_data_span2_begin = data_span2_begin
        temp_data_span2_end = data_span2_end
    else:
        temp_data_span1_begin = data_span1_begin
        temp_data_span1_end = data_span1_end
        temp_data_span2_begin = data_span2_begin - len(temp_seg)
        temp_data_span2_end = data_span2_end - len(temp_seg)
    return fact, temp_data_span1_begin, temp_data_span1_end, temp_data_span2_begin, temp_data_span2_end

def gen_new_sentence(data_path):
    belong_list = []
    str_list = []
    poi_list = []
    fact_list = []
    label_list = []
    span1_begin_list = []
    span1_end_list = []
    span2_begin_list = []
    span2_end_list = []
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)
    poi_arr = np.load(os.path.join(data_path, 'poi_arr.npy'), allow_pickle=True)
    for data_num in range(len(seg_arr)):
        ori_fact = test_data.loc[data_num, 'text']
        label = test_data.loc[data_num, 'label_id']
        data_span1_begin = test_data.loc[data_num, 'span1_begin']
        data_span1_end = test_data.loc[data_num, 'span1_end']
        data_span2_begin = test_data.loc[data_num, 'span2_begin']
        data_span2_end = test_data.loc[data_num, 'span2_end']
        data_segs = seg_arr[data_num]
        positions = poi_arr[data_num]
        # print(data_segs)
        # print(positions)
        for seg_num in range(len(data_segs)):
            temp_seg = data_segs[seg_num]
            temp_poi = positions[seg_num]
            if temp_seg == ' ':
                continue
            temp_fact, temp_span1_begin, temp_span1_end, temp_span2_begin, temp_span2_end = get_new_data(ori_fact, temp_seg, temp_poi, data_span1_begin, data_span1_end, data_span2_begin, data_span2_end)
            belong_list.append(data_num)
            str_list.append(temp_seg)
            poi_list.append(temp_poi)
            fact_list.append(temp_fact)
            label_list.append(label)
            span1_begin_list.append(temp_span1_begin)
            span1_end_list.append(temp_span1_end)
            span2_begin_list.append(temp_span2_begin)
            span2_end_list.append(temp_span2_end)

    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'position': poi_list, 'text': fact_list, 'label_id': label_list,
                     'span1_begin': span1_begin_list, 'span1_end': span1_end_list, 'span2_begin': span2_begin_list,
                     'span2_end': span2_end_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)
    return data_df

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def cal_word_importance(save_path, base_model_path):
    segmentation(save_path)
    new_data_df = gen_new_sentence(save_path)

    config = Config()
    config.pretrained_model = base_model_path

    temp_data_dataloader = get_temp_data(new_data_df, config.max_length, config.batch_size)
    model = BERT(config)
    model = model.to(config.device)
    model.eval()
    test_flag = 0
    with torch.no_grad():
        for _, data in enumerate(temp_data_dataloader, 0):
            print(_)
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
            outputs = model(input)
            outputs = outputs.cpu().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = outputs
            else:
                total_npy = np.concatenate((total_npy, outputs), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_list = np.array(output_list)
    source_data_output = np.load(os.path.join(save_path, 'testing_data_output.npy'))
    importance_value = []
    for data_num in range(len(output_list)):
        temp_output = output_list[data_num]
        temp_label = new_data_df.loc[data_num, 'label_id']
        temp_value = source_data_output[new_data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    new_data_df['importance_value'] = importance_value
    df_filtered = new_data_df[new_data_df['importance_value'] > -100].reset_index(drop=True)
    df_filtered.to_csv(os.path.join(save_path, 'temp_data.csv'), index=False)
    seg_list = []
    seg_position = []
    seg_score = []
    for class_num in range(len(source_data_output)):
        seg_list.append([])
        seg_score.append([])
        seg_position.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
        seg_position[class_name].append(df_filtered.loc[df_filtered_num, 'position'])
    # print(seg_list)
    # print(seg_score)
    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)
    seg_position_arr = np.array(seg_position)
    np.save(os.path.join(save_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(save_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))
    np.save(os.path.join(save_path, 'seg_position.npy'), seg_position_arr)
    print(len(seg_position_arr))



if __name__ == '__main__':
    pass
