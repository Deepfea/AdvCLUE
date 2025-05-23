import numpy as np
import os
import pandas as pd
import torch
import transformers

from method.step1_adv_sample_generation.CHID.seg_data_to_words import seg_data
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
from method.step1_adv_sample_generation.CHID.create_dataset import ClozeDataset
from tqdm import tqdm
import re

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def gen_new_sentence(data_path):
    belong_list = []
    str_list = []
    fact_list = []
    tag_list = []
    label_list = []
    groundTruth_list = []
    data_candidates_list = []
    data_candidate_ids_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    idiom_list = ['#idiom1#', '#idiom2#', '#idiom3#', '#idiom4#', '#idiom5#']
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_candidates = np.load(os.path.join(data_path, 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(data_path, 'test_candidate_ids.npy'), allow_pickle=True)

    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)

    for data_num in range(len(seg_arr)):
        temp_segs = seg_arr[data_num]
        fact = test_data.loc[data_num, 'text']
        tag = test_data.loc[data_num, 'tag']
        label = test_data.loc[data_num, 'label']
        groundTruth = test_data.loc[data_num, 'groundTruth']
        data_candidates = test_candidates[data_num]
        data_candidate_ids = test_candidate_ids[data_num]

        for seg_num in range(len(temp_segs)):
            temp_seg = temp_segs[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            flag = 0
            for idiom_num in range(len(idiom_list)):
                temp_idiom = idiom_list[idiom_num]
                if temp_seg in temp_idiom:
                    flag = 1
                    break
            if flag == 1:
                continue
            temp_fact = fact.replace(temp_seg, '')

            belong_list.append(data_num)
            str_list.append(temp_seg)
            fact_list.append(temp_fact)
            tag_list.append(tag)
            label_list.append(label)
            groundTruth_list.append(groundTruth)
            data_candidates_list.append(data_candidates)
            data_candidate_ids_list.append(data_candidate_ids)


    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'text': fact_list, 'tag': tag_list,
                     'groundTruth': groundTruth_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_candidates_arr = np.array(data_candidates_list)
    data_candidate_ids_arr = np.array(data_candidate_ids_list)

    return data_df, data_candidates_arr, data_candidate_ids_arr

def convert_to_features(df_data, candidates, candidate_ids):
    dataset = ClozeDataset(df_data, candidates, candidate_ids)
    datas = []
    data = []
    batch_id = 1

    for bi, item in enumerate(tqdm(dataset, total=len(dataset))):
        data.append(item)
        if len(data) == 50000 or bi == len(dataset) - 1:
            batch_id += 1
            datas.extend(data)
            data = []
    dataset = datas

    return dataset

def run_one_step(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    input_mask = batch["input_masks"].to(device)
    position = batch["position"].to(device)
    idiom_ids = batch["idiom_ids"].to(device)
    logits = model(
        input_ids,
        input_mask,
        token_type_ids=token_type_ids,
        idiom_ids=idiom_ids,
        positions=position,
    )  #

    return logits

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def cal_word_importance(save_path, base_model_path):
    config = Config()
    data_path = os.path.join(save_path, 'source_data')
    segmentation(data_path)

    new_data_df, new_data_candidates_arr, new_data_candidate_ids_arr = gen_new_sentence(data_path)
    # print(new_data_df.loc[0])
    new_data_dataset = convert_to_features(new_data_df, new_data_candidates_arr, new_data_candidate_ids_arr)
    new_data_dataloader = DataLoader(new_data_dataset, batch_size=config.batch_size, shuffle=False)

    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    trained_model_path = os.path.join(base_model_path, "pytorch_model.bin")

    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

    model.eval()
    output_list = []

    with torch.no_grad():
        tk0 = tqdm(new_data_dataloader, total=len(new_data_dataloader))

        for bi, batch in enumerate(tk0):
            logits = run_one_step(batch, model, config.device)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            outputs = list(outputs)
            output_list.extend(outputs)

    output_arr = np.array(output_list)

    source_data_output = np.load(os.path.join(data_path, 'testing_data_output.npy'))

    importance_value = []
    for data_num in range(len(output_arr)):
        temp_output = np.array(output_arr[data_num])
        temp_label = new_data_df.loc[data_num, 'label']
        temp_value = source_data_output[new_data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    new_data_df['importance_value'] = importance_value
    print(len(new_data_df))
    df_filtered = new_data_df[new_data_df['importance_value'] > -100].reset_index(drop=True)
    print(len(df_filtered))
    # df_filtered = new_data_df
    df_filtered.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)
    # np.save(os.path.join(data_path, 'temp_data_candidates.npy'), new_data_candidates_arr)
    # np.save(os.path.join(data_path, 'temp_data_candidate_ids.npy'), new_data_candidate_ids_arr)

    seg_list = []
    seg_score = []
    for class_num in range(len(source_data_output)):
        seg_list.append([])
        seg_score.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
    print(seg_list)
    print(seg_score)

    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)

    np.save(os.path.join(data_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(data_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))

if __name__ == '__main__':
    pass
