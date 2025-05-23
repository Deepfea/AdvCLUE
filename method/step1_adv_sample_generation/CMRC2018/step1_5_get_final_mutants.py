import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset
from method.step1_adv_sample_generation.CMRC2018.load_token import get_token
from transformers import BertForQuestionAnswering


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

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

def cal_candidate_output(base_model_path, load_path, rate):


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
        print(os.path.join(load_data_path, load_name_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_data_path, load_name_list[name_num]))
        labels = cal_output_label(candidate_mutants, base_model_path)
        # print(labels)
        np.save(os.path.join(load_data_path, label_file_list[name_num]), labels)

def select_final_mutants(save_path, model_path, source_data_path, rates):
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

    test_data_label = np.load(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'))
    for name_num in range(len(mutant_candidate_list)):
        print(mutant_candidate_list[name_num])
        belong_list = []
        ori_text_a_list = []
        ori_text_b_list = []
        ori_answer_start_list = []
        ori_answer_end_list = []
        ori_answer_list = []
        text_a_list = []
        text_b_list = []
        answer_start_list = []
        answer_end_list = []
        str_list = []
        word_list = []
        text_list = []
        pre_list = []
        mut_list = []
        candidates_1 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[0]), mutant_candidate_list[name_num]))
        pre_label_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), label_file_list[name_num]))

        candidates_2 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[1]), mutant_candidate_list[name_num]))
        pre_label_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), label_file_list[name_num]))

        candidates_3 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[2]), mutant_candidate_list[name_num]))
        pre_label_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), label_file_list[name_num]))

        for data_num in tqdm(range(len(test_data_label))):
            if int(test_data_label[data_num][0]) != int(pre_label_1[data_num][0]) and int(test_data_label[data_num][1]) != int(pre_label_1[data_num][1]):
                belong_list.append(data_num)
                ori_text_a_list.append(candidates_1.loc[data_num, 'ori_context'])
                ori_text_b_list.append(candidates_1.loc[data_num, 'ori_question'])
                ori_answer_start_list.append(candidates_1.loc[data_num, 'ori_answer_start'])
                ori_answer_end_list.append(candidates_1.loc[data_num, 'ori_answer_end'])
                ori_answer_list.append(candidates_1.loc[data_num, 'answer'])
                text_a_list.append(candidates_1.loc[data_num, 'context'])
                text_b_list.append(candidates_1.loc[data_num, 'question'])
                answer_start_list.append(candidates_1.loc[data_num, 'answer_start'])
                answer_end_list.append(candidates_1.loc[data_num, 'answer_end'])
                str_list.append(candidates_1.loc[data_num, 'str'])
                word_list.append(candidates_1.loc[data_num, 'word'])
                text_list.append(candidates_1.loc[data_num, 'text'])
                pre_list.append(pre_label_1[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num][0]) != int(pre_label_2[data_num][0]) and int(
                    test_data_label[data_num][1]) != int(pre_label_2[data_num][1]):
                belong_list.append(data_num)
                ori_text_a_list.append(candidates_2.loc[data_num, 'ori_context'])
                ori_text_b_list.append(candidates_2.loc[data_num, 'ori_question'])
                ori_answer_start_list.append(candidates_2.loc[data_num, 'ori_answer_start'])
                ori_answer_end_list.append(candidates_2.loc[data_num, 'ori_answer_end'])
                ori_answer_list.append(candidates_2.loc[data_num, 'answer'])
                text_a_list.append(candidates_2.loc[data_num, 'context'])
                text_b_list.append(candidates_2.loc[data_num, 'question'])
                answer_start_list.append(candidates_2.loc[data_num, 'answer_start'])
                answer_end_list.append(candidates_2.loc[data_num, 'answer_end'])
                str_list.append(candidates_2.loc[data_num, 'str'])
                word_list.append(candidates_2.loc[data_num, 'word'])
                text_list.append(candidates_2.loc[data_num, 'text'])
                pre_list.append(pre_label_2[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num][0]) != int(pre_label_3[data_num][0]) and int(
                    test_data_label[data_num][1]) != int(pre_label_3[data_num][1]):
                belong_list.append(data_num)
                ori_text_a_list.append(candidates_3.loc[data_num, 'ori_context'])
                ori_text_b_list.append(candidates_3.loc[data_num, 'ori_question'])
                ori_answer_start_list.append(candidates_3.loc[data_num, 'ori_answer_start'])
                ori_answer_end_list.append(candidates_3.loc[data_num, 'ori_answer_end'])
                ori_answer_list.append(candidates_3.loc[data_num, 'answer'])
                text_a_list.append(candidates_3.loc[data_num, 'context'])
                text_b_list.append(candidates_3.loc[data_num, 'question'])
                answer_start_list.append(candidates_3.loc[data_num, 'answer_start'])
                answer_end_list.append(candidates_3.loc[data_num, 'answer_end'])
                str_list.append(candidates_3.loc[data_num, 'str'])
                word_list.append(candidates_3.loc[data_num, 'word'])
                text_list.append(candidates_3.loc[data_num, 'text'])
                pre_list.append(pre_label_3[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
        merge_dt_dict = {'belong': belong_list, 'ori_context': ori_text_a_list, 'ori_question': ori_text_b_list, 'ori_answer_start': ori_answer_start_list,
                         'ori_answer_end': ori_answer_end_list, 'answer': ori_answer_list, 'context': text_a_list, 'question': text_b_list,
                         'answer_start': answer_start_list, 'answer_end': answer_end_list, 'str': str_list, 'word':word_list,
                         'text': text_list, 'pre': pre_list, 'mut': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(save_path, final_mutants_list[name_num]), index=False)
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
