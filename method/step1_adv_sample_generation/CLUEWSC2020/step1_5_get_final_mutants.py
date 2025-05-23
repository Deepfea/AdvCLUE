import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from method.step1_adv_sample_generation.TNEWS.create_mutant import MyMutant
from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset
from dataset import get_dev_dataloader, get_temp_data
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def cal_output_label(test_data, base_model_path, tokenizer_path):
    config = Config()
    config.pretrained_model = base_model_path
    dev_dataloader = get_temp_data(test_data, config.max_length, config.batch_size)
    model = BERT(config)
    model = model.to(config.device)
    model.eval()
    test_flag = 0
    with torch.no_grad():
        for _, data in enumerate(dev_dataloader, 0):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
            outputs = model(input)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = outputs
                total_label = targets
            else:
                total_npy = np.concatenate((total_npy, outputs), axis=0)
                total_label = np.concatenate((total_label, targets), axis=0)
    predict_label = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_label = np.argmax(temp_npy)
        predict_label.append(temp_label)
    predict_label = np.array(predict_label)
    return predict_label

def cal_candidate_output(base_model_path, tokenizer_path, load_path, rate):
    test_data_path = os.path.join(load_path, 'source_data', 'test.csv')
    test_data = pd.read_csv(test_data_path)
    # print(test_data)
    labels = cal_output_label(test_data, base_model_path, tokenizer_path)
    np.save(os.path.join(load_path, 'source_data', 'predict_lables.npy'), labels)

    load_data_path = os.path.join(load_path, 'source_data', str(rate))
    load_name_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                      'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                      'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                      'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                      'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                      'traditional_conversion_candidate_mutants.csv']
    label_file_list = ['word_shuffling_labels.npy', 'character_deleting_labels.npy',
                       'symbol_insertion_labels.npy', 'glyph_replacement_labels.npy',
                       'character_splitting_labels.npy', 'homophone_replacement_labels.npy',
                       'nasal_replacement_labels.npy', 'dorsal_replacement_labels.npy',
                       'context_prediction_labels.npy', 'synonym_replacement_labels.npy',
                       'traditional_conversion_labels.npy']
    for name_num in range(len(load_name_list)):
        candidate_mutants = pd.read_csv(os.path.join(load_data_path, load_name_list[name_num]))
        labels = cal_output_label(candidate_mutants, base_model_path, tokenizer_path)
        np.save(os.path.join(load_data_path, label_file_list[name_num]), labels)

def select_final_mutants(save_path, tokenizers_path, model_path, source_data_path, rates):
    mutant_candidate_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                        'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                        'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                        'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                        'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                        'traditional_conversion_candidate_mutants.csv']
    final_mutants_list = ['word_shuffling_mutants.csv', 'character_deleting_mutants.csv',
                        'symbol_insertion_mutants.csv', 'glyph_replacement_mutants.csv',
                        'character_splitting_mutants.csv', 'homophone_replacement_mutants.csv',
                        'nasal_replacement_mutants.csv', 'dorsal_replacement_mutants.csv',
                        'context_prediction_mutants.csv', 'synonym_replacement_mutants.csv',
                        'traditional_conversion_mutants.csv']
    label_file_list = ['word_shuffling_labels.npy', 'character_deleting_labels.npy',
                       'symbol_insertion_labels.npy', 'glyph_replacement_labels.npy',
                       'character_splitting_labels.npy', 'homophone_replacement_labels.npy',
                       'nasal_replacement_labels.npy', 'dorsal_replacement_labels.npy',
                       'context_prediction_labels.npy', 'synonym_replacement_labels.npy',
                       'traditional_conversion_labels.npy']
    mut_name_list = ['word_shuffling', 'character_deleting',
                       'symbol_insertion', 'glyph_replacement',
                       'character_splitting', 'homophone_replacement',
                       'nasal_replacement', 'dorsal_replacement',
                       'context_prediction', 'synonym_replacement',
                       'traditional_conversion']
    # for rate in rates:
    #     cal_candidate_output(model_path, tokenizers_path, source_data_path, rate)
    test_data_label = np.load(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'))
    for name_num in range(len(mutant_candidate_list)):
        print(mutant_candidate_list[name_num])
        belong_list = []
        fact_list = []
        mutant_list = []
        str_list = []
        word_list = []
        label_list = []
        pre_list = []
        mut_list = []
        span1_begin_list = []
        span1_end_list = []
        span2_begin_list = []
        span2_end_list = []
        candidates_1 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[0]), mutant_candidate_list[name_num]))
        pre_label_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), label_file_list[name_num]))

        candidates_2 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[1]), mutant_candidate_list[name_num]))
        pre_label_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), label_file_list[name_num]))

        candidates_3 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[2]), mutant_candidate_list[name_num]))
        pre_label_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), label_file_list[name_num]))

        for data_num in tqdm(range(len(test_data_label))):
            if int(test_data_label[data_num]) != int(pre_label_1[data_num]):
                belong_list.append(data_num)
                fact_list.append(candidates_1.loc[data_num, 'fact'])
                mutant_list.append(candidates_1.loc[data_num, 'text'])
                str_list.append(candidates_1.loc[data_num, 'str'])
                word_list.append(candidates_1.loc[data_num, 'word'])
                label_list.append(candidates_1.loc[data_num, 'label_id'])
                span1_begin_list.append(candidates_1.loc[data_num, 'span1_begin'])
                span1_end_list.append(candidates_1.loc[data_num, 'span1_end'])
                span2_begin_list.append(candidates_1.loc[data_num, 'span2_begin'])
                span2_end_list.append(candidates_1.loc[data_num, 'span2_end'])
                pre_list.append(pre_label_1[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_2[data_num]):
                belong_list.append(data_num)
                fact_list.append(candidates_2.loc[data_num, 'fact'])
                mutant_list.append(candidates_2.loc[data_num, 'text'])
                str_list.append(candidates_2.loc[data_num, 'str'])
                word_list.append(candidates_2.loc[data_num, 'word'])
                label_list.append(candidates_2.loc[data_num, 'label_id'])
                span1_begin_list.append(candidates_2.loc[data_num, 'span1_begin'])
                span1_end_list.append(candidates_2.loc[data_num, 'span1_end'])
                span2_begin_list.append(candidates_2.loc[data_num, 'span2_begin'])
                span2_end_list.append(candidates_2.loc[data_num, 'span2_end'])
                pre_list.append(pre_label_2[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_3[data_num]):
                belong_list.append(data_num)
                fact_list.append(candidates_3.loc[data_num, 'fact'])
                mutant_list.append(candidates_3.loc[data_num, 'text'])
                str_list.append(candidates_3.loc[data_num, 'str'])
                word_list.append(candidates_3.loc[data_num, 'word'])
                label_list.append(candidates_3.loc[data_num, 'label_id'])
                span1_begin_list.append(candidates_3.loc[data_num, 'span1_begin'])
                span1_end_list.append(candidates_3.loc[data_num, 'span1_end'])
                span2_begin_list.append(candidates_3.loc[data_num, 'span2_begin'])
                span2_end_list.append(candidates_3.loc[data_num, 'span2_end'])
                pre_list.append(pre_label_3[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
        merge_dt_dict = {'belong': belong_list, 'fact': fact_list, 'str': str_list, 'word': word_list, 'text': mutant_list,
                         'span1_begin': span1_begin_list, 'span1_end': span1_end_list,
                         'span2_begin': span2_begin_list, 'span2_end': span2_end_list,
                         'label_id': label_list, 'prediction': pre_list, 'type': mut_list}

        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(save_path, final_mutants_list[name_num]), index=False)
        print(len(data_df))
    flag = 0
    for mut_num in range(len(final_mutants_list)):
        mutant_path = os.path.join(save_path, final_mutants_list[mut_num])
        temp_mutants = pd.read_csv(mutant_path)
        if flag == 0:
            final_mutants = temp_mutants
            flag = 1
        else:
            final_mutants = pd.concat([final_mutants, temp_mutants], ignore_index=True)
    if flag == 0:
        print("一个变异成功的数据也没有！！！")
    else:
        print(len(final_mutants))
        final_mutants.to_csv(os.path.join(save_path, 'final_mutants.csv'), index=False)


if __name__ == '__main__':
    pass
