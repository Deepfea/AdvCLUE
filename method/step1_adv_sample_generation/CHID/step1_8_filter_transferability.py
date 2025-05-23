import os
import pandas as pd
from bert_score import score
from torch.utils.data import DataLoader
from tqdm import tqdm
from method.step1_adv_sample_generation.CHID.create_dataset_roberta import ClozeDataset
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
import numpy as np
import os
import torch
from transformers import BertTokenizer
import pandas as pd
import transformers


fidelity_mutants_list = ['fidelity_word_shuffling_mutants.csv', 'fidelity_character_deleting_mutants.csv',
                      'fidelity_symbol_insertion_mutants.csv', 'fidelity_glyph_replacement_mutants.csv',
                      'fidelity_character_splitting_mutants.csv', 'fidelity_homophone_replacement_mutants.csv',
                      'fidelity_nasal_replacement_mutants.csv', 'fidelity_dorsal_replacement_mutants.csv',
                      'fidelity_context_prediction_mutants.csv', 'fidelity_synonym_replacement_mutants.csv',
                      'fidelity_traditional_conversion_mutants.csv']
fidelity_mut_name_list = ['fidelity_word_shuffling', 'fidelity_character_deleting',
                       'fidelity_symbol_insertion', 'fidelity_glyph_replacement',
                       'fidelity_character_splitting', 'fidelity_homophone_replacement',
                       'fidelity_nasal_replacement', 'fidelity_dorsal_replacement',
                       'fidelity_context_prediction', 'fidelity_synonym_replacement',
                       'fidelity_traditional_conversion']
transferability_mutants_list = ['transferability_word_shuffling_mutants.csv', 'transferability_character_deleting_mutants.csv',
                      'transferability_symbol_insertion_mutants.csv', 'transferability_glyph_replacement_mutants.csv',
                      'transferability_character_splitting_mutants.csv', 'transferability_homophone_replacement_mutants.csv',
                      'transferability_nasal_replacement_mutants.csv', 'transferability_dorsal_replacement_mutants.csv',
                      'transferability_context_prediction_mutants.csv', 'transferability_synonym_replacement_mutants.csv',
                      'transferability_traditional_conversion_mutants.csv']
transferability_mut_name_list = ['transferability_word_shuffling', 'transferability_character_deleting',
                       'transferability_symbol_insertion', 'transferability_glyph_replacement',
                       'transferability_character_splitting', 'transferability_homophone_replacement',
                       'transferability_nasal_replacement', 'transferability_dorsal_replacement',
                       'transferability_context_prediction', 'transferability_synonym_replacement',
                       'transferability_traditional_conversion']

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese'
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese"

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

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

def cal_output_label(test_df, test_candidates, test_candidate_ids, base_model_path):
    config = Config()
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                     config=model_config, idiom_num=len(idiom_vocab))
    trained_model_path = os.path.join(base_model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    model.eval()
    output_list = []
    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))

        for bi, batch in enumerate(tk0):
            logits = run_one_step(batch, model, config.device)
            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            outputs = list(outputs)
            output_list.extend(outputs)
    output_arr = np.array(output_list)
    pre_labels = []
    for num in range(len(output_arr)):
        temp_arr = output_arr[num]
        pre_labels.append(np.argmax(temp_arr))
    output_arr = np.array(pre_labels)
    return output_arr

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese')
    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        test_candidates = np.load(os.path.join(load_path, fidelity_mut_name_list[name_num] + '_mutant_candidate.npy'),
                                  allow_pickle=True)
        test_candidate_ids = np.load(os.path.join(load_path, fidelity_mut_name_list[name_num] + '_mutant_candidate_ids.npy'),
                                     allow_pickle=True)
        pre_labels = cal_output_label(candidate_mutants, test_candidates, test_candidate_ids, model_path)
        true_labels = list(candidate_mutants['label'])

        belong_list = []
        tag_list = []
        ori_text_list = []
        str_list = []
        word_list = []
        text_list = []
        candidate_list = []
        groundTruth_list = []
        label_list = []
        mut_list = []
        mutant_candidates = []
        mutant_candidate_ids = []

        for data_num in range(len(true_labels)):
            if int(pre_labels[data_num]) == int(true_labels[data_num]):
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            tag_list.append(candidate_mutants.loc[data_num, 'tag'])
            ori_text_list.append(candidate_mutants.loc[data_num, 'ori_text'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            candidate_list.append(candidate_mutants.loc[data_num, 'candidate'])
            groundTruth_list.append(candidate_mutants.loc[data_num, 'groundTruth'])
            label_list.append(candidate_mutants.loc[data_num, 'label'])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])
            mutant_candidates.append(test_candidates[data_num])
            mutant_candidate_ids.append(test_candidate_ids[data_num])
        merge_dt_dict = {'belong': belong_list, 'tag': tag_list, 'ori_text': ori_text_list,
                         'str': str_list, 'word': word_list, 'text': text_list,
                         'candidate': candidate_list, 'groundTruth': groundTruth_list, 'label': label_list,
                         'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

        mutant_candidates = np.array(mutant_candidates)
        mutant_candidate_ids = np.array(mutant_candidate_ids)

        np.save(os.path.join(load_path, transferability_mut_name_list[name_num] + '_mutant_candidate.npy'),
                mutant_candidates)
        np.save(os.path.join(load_path, transferability_mut_name_list[name_num] + '_mutant_candidate_ids.npy'),
                mutant_candidate_ids)
        print(len(mutant_candidates))


if __name__ == '__main__':
    pass
