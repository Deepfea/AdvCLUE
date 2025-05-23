import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from method.step1_adv_sample_generation.CHID.create_dataset import ClozeDataset
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
from transformers import BertTokenizer
import transformers

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese'
    # BERT_PATH = project_dir + "/pretrained_models/ernie_based_pretrained"
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

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

def cal_candidate_output(test_candidates, test_candidate_ids, base_model_path, load_path, rate):
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
        labels = cal_output_label(candidate_mutants, test_candidates, test_candidate_ids, base_model_path)
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
    test_data_path = os.path.join(source_data_path, 'source_data', 'test.csv')
    test_data = pd.read_csv(test_data_path)
    test_candidates = np.load(os.path.join(source_data_path, 'source_data', 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(source_data_path, 'source_data', 'test_candidate_ids.npy'), allow_pickle=True)

    # print(test_data)
    # labels = cal_output_label(test_data, test_candidates, test_candidate_ids, model_path)
    # np.save(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'), labels)
    # print(labels)

    # for rate in rates:
    #     cal_candidate_output(test_candidates, test_candidate_ids, model_path, source_data_path, rate)
    test_data_label = np.load(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'))
    for name_num in range(len(mutant_candidate_list)):
        print(mutant_candidate_list[name_num])
        belong_list = []
        data_id_list = []
        tag_list = []
        ori_text_list = []
        text_list = []
        str_list = []
        word_list = []
        candidate_list = []
        groundTruth_list = []
        label_list = []
        pre_list = []
        mut_list = []
        mutant_candidates = []
        mutant_candidate_ids = []

        candidates_1 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[0]), mutant_candidate_list[name_num]))
        pre_label_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), label_file_list[name_num]))

        candidates_2 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[1]), mutant_candidate_list[name_num]))
        pre_label_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), label_file_list[name_num]))

        candidates_3 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[2]), mutant_candidate_list[name_num]))
        pre_label_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), label_file_list[name_num]))

        for data_num in tqdm(range(len(test_data_label))):
            if int(test_data_label[data_num]) != int(pre_label_1[data_num]):
                belong_list.append(data_num)
                data_id_list.append(candidates_1.loc[data_num, 'data_id'])
                tag_list.append(candidates_1.loc[data_num, 'tag'])
                ori_text_list.append(candidates_1.loc[data_num, 'ori_text'])
                text_list.append(candidates_1.loc[data_num, 'text'])
                str_list.append(candidates_1.loc[data_num, 'str'])
                word_list.append(candidates_1.loc[data_num, 'word'])
                candidate_list.append(candidates_1.loc[data_num, 'candidate'])
                groundTruth_list.append(candidates_1.loc[data_num, 'groundTruth'])
                label_list.append(candidates_1.loc[data_num, 'label'])
                pre_list.append(pre_label_1[data_num])
                mut_list.append(mut_name_list[name_num])
                mutant_candidates.append(test_candidates[data_num])
                mutant_candidate_ids.append(test_candidate_ids[data_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_2[data_num]):
                belong_list.append(data_num)
                data_id_list.append(candidates_2.loc[data_num, 'data_id'])
                tag_list.append(candidates_2.loc[data_num, 'tag'])
                ori_text_list.append(candidates_2.loc[data_num, 'ori_text'])
                text_list.append(candidates_2.loc[data_num, 'text'])
                str_list.append(candidates_2.loc[data_num, 'str'])
                word_list.append(candidates_2.loc[data_num, 'word'])
                candidate_list.append(candidates_2.loc[data_num, 'candidate'])
                groundTruth_list.append(candidates_2.loc[data_num, 'groundTruth'])
                label_list.append(candidates_2.loc[data_num, 'label'])
                pre_list.append(pre_label_2[data_num])
                mut_list.append(mut_name_list[name_num])
                mutant_candidates.append(test_candidates[data_num])
                mutant_candidate_ids.append(test_candidate_ids[data_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_3[data_num]):
                belong_list.append(data_num)
                data_id_list.append(candidates_3.loc[data_num, 'data_id'])
                tag_list.append(candidates_3.loc[data_num, 'tag'])
                ori_text_list.append(candidates_3.loc[data_num, 'ori_text'])
                text_list.append(candidates_3.loc[data_num, 'text'])
                str_list.append(candidates_3.loc[data_num, 'str'])
                word_list.append(candidates_3.loc[data_num, 'word'])
                candidate_list.append(candidates_3.loc[data_num, 'candidate'])
                groundTruth_list.append(candidates_3.loc[data_num, 'groundTruth'])
                label_list.append(candidates_3.loc[data_num, 'label'])
                pre_list.append(pre_label_3[data_num])
                mut_list.append(mut_name_list[name_num])
                mutant_candidates.append(test_candidates[data_num])
                mutant_candidate_ids.append(test_candidate_ids[data_num])
        merge_dt_dict = {'belong': belong_list, 'data_id': data_id_list, 'tag': tag_list,
                         'ori_text': ori_text_list, 'text': text_list, 'str': str_list, 'word': word_list,
                         'candidate': candidate_list, 'groundTruth': groundTruth_list,
                         'label': label_list, 'pre': pre_list, 'mut': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(save_path, final_mutants_list[name_num]), index=False)
        mutant_candidates = np.array(mutant_candidates)
        mutant_candidate_ids = np.array(mutant_candidate_ids)
        np.save(os.path.join(save_path, mut_name_list[name_num] + '_mutant_candidate.npy'), mutant_candidates)
        np.save(os.path.join(save_path, mut_name_list[name_num] + '_mutant_candidate_ids.npy'), mutant_candidate_ids)

        print(len(data_df))
        print(len(mutant_candidates))


if __name__ == '__main__':
    pass
