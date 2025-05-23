import numpy as np
import os

import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertForQuestionAnswering
from method.step1_adv_sample_generation.CMRC2018.load_token_roberta import get_token
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset
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
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese"

def cal_output_label(new_data_df, base_model_path):
    config = Config()
    new_data_encodings = get_token(new_data_df)
    new_data_dataset = SquadDataset(new_data_encodings, config)
    new_data_loader = DataLoader(new_data_dataset, batch_size=config.batch_size, shuffle=False)
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    model.eval()
    start_list = []
    end_list = []
    for idx, batch in enumerate(tqdm(new_data_loader)):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        temp_start = outputs.start_logits.detach().cpu().clone().numpy()
        start_list.extend(temp_start)
        temp_end = outputs.end_logits.detach().cpu().clone().numpy()
        end_list.extend(temp_end)
    start_labels = []
    end_labels = []
    for num in range(len(start_list)):
        start_index = np.argmax(start_list[num])
        start_labels.append(start_index)
        end_index = np.argmax(end_list[num])
        end_labels.append(end_index)
    start_labels = torch.tensor(start_labels)
    end_labels = torch.tensor(end_labels)
    result = []
    for num in range(len(start_labels)):
        result.append([])
        start_pre = new_data_encodings.token_to_chars(num, start_labels[num])
        end_pre = new_data_encodings.token_to_chars(num, end_labels[num])
        if start_pre == None:
            result[num].append(-1)
        else:
            result[num].append(start_pre.start)
        if end_pre == None:
            result[num].append(-1)
        else:
            result[num].append(end_pre.end)

    return result

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese', 'best_CMRC2018_roberta_base_chinese.model')

    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        pre_labels = cal_output_label(candidate_mutants, model_path)

        answer_starts = list(candidate_mutants['answer_start'])
        answer_ends = list(candidate_mutants['answer_end'])

        belong_list = []
        ori_context_list = []
        ori_question_list = []
        ori_answer_start_list = []
        ori_answer_end_list = []
        answer_list = []
        context_list = []
        question_list = []
        answer_start_list = []
        answer_end_list = []
        str_list = []
        word_list = []
        text_list = []
        mut_list = []

        for data_num in range(len(pre_labels)):
            if int(pre_labels[data_num][0]) == int(answer_starts[data_num]) and int(pre_labels[data_num][1]) == int(answer_ends[data_num]):
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            ori_context_list.append(candidate_mutants.loc[data_num, 'ori_context'])
            ori_question_list.append(candidate_mutants.loc[data_num, 'ori_question'])
            ori_answer_start_list.append(candidate_mutants.loc[data_num, 'ori_answer_start'])
            ori_answer_end_list.append(candidate_mutants.loc[data_num, 'ori_answer_end'])
            answer_list.append(candidate_mutants.loc[data_num, 'answer'])
            context_list.append(candidate_mutants.loc[data_num, 'context'])
            question_list.append(candidate_mutants.loc[data_num, 'question'])
            answer_start_list.append(candidate_mutants.loc[data_num, 'answer_start'])
            answer_end_list.append(candidate_mutants.loc[data_num, 'answer_end'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])

        merge_dt_dict = {'belong': belong_list, 'ori_context': ori_context_list, 'ori_question': ori_question_list,
                         'ori_answer_start': ori_answer_start_list, 'ori_answer_end': ori_answer_end_list, 'answer': answer_list,
                         'context': context_list, 'question': question_list, 'answer_start': answer_start_list,
                         'answer_end': answer_end_list, 'str': str_list, 'word': word_list,
                         'text': text_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
