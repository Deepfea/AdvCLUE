import os
import pandas as pd
from bert_score import score
import numpy as np
import os
import torch
from transformers import BertTokenizer
import pandas as pd
from dataset import get_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT

fidelity_mutants_list = ['fidelity_word_shuffling_mutants.csv', 'fidelity_character_deleting_mutants.csv',
                      'fidelity_symbol_insertion_mutants.csv', 'fidelity_glyph_replacement_mutants.csv',
                      'fidelity_character_splitting_mutants.csv', 'fidelity_homophone_replacement_mutants.csv',
                      'fidelity_nasal_replacement_mutants.csv', 'fidelity_dorsal_replacement_mutants.csv',
                      'fidelity_context_prediction_mutants.csv', 'fidelity_synonym_replacement_mutants.csv',
                      'fidelity_traditional_conversion_mutants.csv']
transferability_mutants_list = ['transferability_word_shuffling_mutants.csv', 'transferability_character_deleting_mutants.csv',
                      'transferability_symbol_insertion_mutants.csv', 'transferability_glyph_replacement_mutants.csv',
                      'transferability_character_splitting_mutants.csv', 'transferability_homophone_replacement_mutants.csv',
                      'transferability_nasal_replacement_mutants.csv', 'transferability_dorsal_replacement_mutants.csv',
                      'transferability_context_prediction_mutants.csv', 'transferability_synonym_replacement_mutants.csv',
                      'transferability_traditional_conversion_mutants.csv']

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def cal_output_label(test_data, base_model_path, tokenizer_path):
    config = Config()
    config.pretrained_model = base_model_path
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    dev_dataloader = get_dataloader(test_data, tokenizer, config.max_length, config.batch_size)
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

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese')
    tokenizer_path = '/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese'
    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        pre_labels = cal_output_label(candidate_mutants, model_path, tokenizer_path)
        true_labels = list(candidate_mutants['label_id'])

        belong_list = []
        ori_text_list = []
        text_list = []
        str_list = []
        word_list = []
        span1_begin_list = []
        span1_end_list = []
        span2_begin_list = []
        span2_end_list = []
        label_id_list = []
        mut_list = []
        for data_num in range(len(true_labels)):
            if int(pre_labels[data_num]) == int(true_labels[data_num]):
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            ori_text_list.append(candidate_mutants.loc[data_num, 'ori_text'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            span1_begin_list.append(candidate_mutants.loc[data_num, 'span1_begin'])
            span1_end_list.append(candidate_mutants.loc[data_num, 'span1_end'])
            span2_begin_list.append(candidate_mutants.loc[data_num, 'span2_begin'])
            span2_end_list.append(candidate_mutants.loc[data_num, 'span2_end'])
            label_id_list.append(candidate_mutants.loc[data_num, 'label_id'])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])

        merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list, 'word': word_list,
                         'text': text_list, 'span1_begin': span1_begin_list, 'span1_end': span1_end_list,
                         'span2_begin': span2_begin_list, 'span2_end': span2_end_list, 'label_id': label_id_list,
                         'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
