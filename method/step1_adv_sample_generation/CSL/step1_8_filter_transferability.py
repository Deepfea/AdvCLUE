import numpy as np
import os
from torch.utils.data import DataLoader
import pandas as pd
from method.step1_adv_sample_generation.CSL.bert_model import BertClassifier
from method.step1_adv_sample_generation.CSL.Data import *
from method.step1_adv_sample_generation.CSL.load_token_roberta import load_token
from tqdm import tqdm

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
    max_length = 510
    batch_size = 16
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese"

def cal_output_label(test_df, base_model_path):
    config = Config()
    model = BertClassifier(config)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)

    test_json = load_token(test_df)
    test_data = SentencePairDataset(test_json, config.max_length, padding_idx=0)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    output_arr = get_output(model, test_loader)
    output_list = []
    for temp_num in range(len(output_arr)):
        temp_npy = output_arr[temp_num]
        temp_label = np.argmax(temp_npy)
        output_list.append(temp_label)
    output_arr = np.array(output_list)
    return output_arr

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
    return np.array(output_list)

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese', 'best_CSL_roberta_base_chinese.model')

    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        pre_labels = cal_output_label(candidate_mutants, model_path)
        true_labels = list(candidate_mutants['label'])

        belong_list = []
        ori_abs_list = []
        ori_keyword_list = []
        abs_list = []
        keyword_list = []
        str_list = []
        word_list = []
        label_list = []
        mut_list = []

        for data_num in range(len(true_labels)):
            if int(pre_labels[data_num]) == int(true_labels[data_num]):
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            ori_abs_list.append(candidate_mutants.loc[data_num, 'ori_abs'])
            ori_keyword_list.append(candidate_mutants.loc[data_num, 'ori_keyword'])
            abs_list.append(candidate_mutants.loc[data_num, 'abs'])
            keyword_list.append(candidate_mutants.loc[data_num, 'keyword'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            label_list.append(candidate_mutants.loc[data_num, 'label'])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])

        merge_dt_dict = {'belong': belong_list, 'ori_abs': ori_abs_list, 'ori_keyword': ori_keyword_list, 'abs': abs_list,
                         'keyword': keyword_list, 'str': str_list, 'word': word_list, 'label': label_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
