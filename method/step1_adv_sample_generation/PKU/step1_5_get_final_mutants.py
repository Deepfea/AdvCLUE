import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from method.step1_adv_sample_generation.PKU.create_PKU_dataset import SegDataset

def get_entities(seq):
    # print(seq)
    prev_tag = 'O'
    seq += ['O']
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq):
        tag = chunk[0]
        # print(tag)
        if end_of_chunk(prev_tag, tag):
            chunks.append((begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    # print(chunks)
    return chunks


def end_of_chunk(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'E':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'M' and tag == 'B':
        chunk_end = True
    if prev_tag == 'M' and tag == 'S':
        chunk_end = True
    if prev_tag == 'M' and tag == 'O':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'O' and tag == 'M':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'M':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'M':
        chunk_start = True
    if prev_tag == 'E' and tag == 'E':
        chunk_start = True

    return chunk_start

def acc_score(y_true, y_pred):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)
    word_acc = float(nb_correct) / float(nb_true)
    return word_acc

def load_dataset(path, file_name):
    dataset_path = os.path.join(path, file_name)
    test_data = np.load(dataset_path, allow_pickle=True)
    data_x = test_data['data']
    data_y = test_data['data_label']
    mutant_x = test_data['mutants']
    mutant_y = test_data['mutants_label']
    ori_strs = test_data['strs']
    mut_strs = test_data['words']

    return data_x, data_y, mutant_x, mutant_y, ori_strs, mut_strs


def evaluate_data(dev_loader, model_path):
    model = torch.load(os.path.join(model_path, 'best_PKU_bert_base_chinese_.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    id2label = {_id: _label for _label, _id in list(label2id.items())}
    true_tags = []
    pred_tags = []
    sent_data = []
    test_output = []
    with torch.no_grad():
        for idx, batch_samples in tqdm(enumerate(dev_loader)):
            batch_data, batch_token_starts, batch_tags, ori_data = batch_samples
            batch_data = batch_data.to(device)
            batch_token_starts = batch_token_starts.to(device)
            batch_tags = batch_tags.to(device)
            sent_data.extend(ori_data)
            batch_masks = batch_data.gt(0)  # get padding mask
            # shape: (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            label_masks = batch_tags.gt(-1).to('cpu').numpy()  # get padding mask
            batch_output = batch_output.detach().cpu().numpy()
            batch_tags = batch_tags.to('cpu').numpy()
            for i, indices in enumerate(np.argmax(batch_output, axis=2)):
                pred_tag = []
                te_output = []
                for j, idx in enumerate(indices):
                    if label_masks[i][j]:
                        pred_tag.append(id2label.get(idx))
                        te_output.append(batch_output[i][j])
                test_output.append(te_output)
                pred_tags.append(pred_tag)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    assert len(sent_data) == len(true_tags)

    return sent_data, pred_tags, true_tags, test_output

def cal_output_label(candidate_mutants, base_model_path):
    x_test = candidate_mutants['mutants']
    y_test = candidate_mutants['mutants_label']

    testing_dataset = SegDataset(x_test, y_test)
    testing_loader = DataLoader(testing_dataset, batch_size=6, collate_fn=testing_dataset.collate_fn)
    sent_data, pred_tags, true_tags, test_output = evaluate_data(testing_loader, base_model_path)
    pred_tags = np.array(pred_tags)
    return pred_tags

def cal_candidate_output(base_model_path, load_path, rate):

    load_data_path = os.path.join(load_path, 'source_data', str(rate))
    load_name_list = ['word_shuffling_candidate_mutants.npz', 'character_deleting_candidate_mutants.npz',
                 'symbol_insertion_candidate_mutants.npz', 'glyph_replacement_candidate_mutants.npz',
                 'character_splitting_candidate_mutants.npz', 'homophone_replacement_candidate_mutants.npz',
                 'nasal_replacement_candidate_mutants.npz', 'dorsal_replacement_candidate_mutants.npz',
                 'context_prediction_candidate_mutants.npz', 'synonym_replacement_candidate_mutants.npz',
                 'traditional_conversion_candidate_mutants.npz']
    label_file_list = ['word_shuffling_labels.npy', 'character_deleting_labels.npy',
                       'symbol_insertion_labels.npy', 'glyph_replacement_labels.npy',
                       'character_splitting_labels.npy', 'homophone_replacement_labels.npy',
                       'nasal_replacement_labels.npy', 'dorsal_replacement_labels.npy',
                       'context_prediction_labels.npy', 'synonym_replacement_labels.npy',
                       'traditional_conversion_labels.npy']
    for name_num in range(len(load_name_list)):
        print(os.path.join(load_data_path, load_name_list[name_num]))
        candidate_mutants = np.load(os.path.join(load_data_path, load_name_list[name_num]), allow_pickle=True)
        labels = cal_output_label(candidate_mutants, base_model_path)
        # print(len(labels))
        np.save(os.path.join(load_data_path, label_file_list[name_num]), labels)

def select_final_mutants(save_path, model_path, source_data_path, rates):
    mutant_candidate_list = ['word_shuffling_candidate_mutants.npz', 'character_deleting_candidate_mutants.npz',
                 'symbol_insertion_candidate_mutants.npz', 'glyph_replacement_candidate_mutants.npz',
                 'character_splitting_candidate_mutants.npz', 'homophone_replacement_candidate_mutants.npz',
                 'nasal_replacement_candidate_mutants.npz', 'dorsal_replacement_candidate_mutants.npz',
                 'context_prediction_candidate_mutants.npz', 'synonym_replacement_candidate_mutants.npz',
                 'traditional_conversion_candidate_mutants.npz']
    final_mutants_list = ['word_shuffling_mutants', 'character_deleting_mutants',
                        'symbol_insertion_mutants', 'glyph_replacement_mutants',
                        'character_splitting_mutants', 'homophone_replacement_mutants',
                        'nasal_replacement_mutants', 'dorsal_replacement_mutants',
                        'context_prediction_mutants', 'synonym_replacement_mutants',
                        'traditional_conversion_mutants']
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
    for rate in rates:
        cal_candidate_output(model_path, source_data_path, rate)
    for name_num in range(len(mutant_candidate_list)):
        print(mutant_candidate_list[name_num])
        belong_list = []
        data_list = []
        data_label_list = []
        mutants_list = []
        mutants_label_list = []
        str_list = []
        word_list = []
        pre_list = []
        mut_list = []

        candidates_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), mutant_candidate_list[name_num]), allow_pickle=True)
        pre_label_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), label_file_list[name_num]), allow_pickle=True)

        candidates_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), mutant_candidate_list[name_num]), allow_pickle=True)
        pre_label_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), label_file_list[name_num]), allow_pickle=True)

        candidates_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), mutant_candidate_list[name_num]), allow_pickle=True)
        pre_label_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), label_file_list[name_num]), allow_pickle=True)

        test_data_x = candidates_1['data']
        test_data_y = candidates_1['data_label']

        for data_num in tqdm(range(len(test_data_x))):
            socre1 = acc_score(candidates_1['mutants_label'][data_num], pre_label_1[data_num])
            if socre1 < 0.85:
                belong_list.append(data_num)
                data_list.append(candidates_1['data'][data_num])
                data_label_list.append(candidates_1['data_label'][data_num])
                mutants_list.append(candidates_1['mutants'][data_num])
                mutants_label_list.append(candidates_1['mutants_label'][data_num])
                str_list.append(candidates_1['strs'][data_num])
                word_list.append((candidates_1['words'][data_num]))
                pre_list.append(pre_label_1[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            score2 = acc_score(candidates_2['mutants_label'][data_num], pre_label_2[data_num])
            if score2 < 0.85:
                belong_list.append(data_num)
                data_list.append(candidates_2['data'][data_num])
                data_label_list.append(candidates_2['data_label'][data_num])
                mutants_list.append(candidates_2['mutants'][data_num])
                mutants_label_list.append(candidates_2['mutants_label'][data_num])
                str_list.append(candidates_2['strs'][data_num])
                word_list.append((candidates_2['words'][data_num]))
                pre_list.append(pre_label_2[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            score3 = acc_score(candidates_3['mutants_label'][data_num], pre_label_3[data_num])
            if score3 < 0.85:
                belong_list.append(data_num)
                data_list.append(candidates_3['data'][data_num])
                data_label_list.append(candidates_3['data_label'][data_num])
                mutants_list.append(candidates_3['mutants'][data_num])
                mutants_label_list.append(candidates_3['mutants_label'][data_num])
                str_list.append(candidates_3['strs'][data_num])
                word_list.append((candidates_3['words'][data_num]))
                pre_list.append(pre_label_3[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
        print(len(mutants_list))
        np.savez_compressed(os.path.join(save_path, final_mutants_list[name_num]), belong=belong_list, data=data_list,
                            data_label=data_label_list, mutants=mutants_list, mutants_label=mutants_label_list,
                            strs=str_list, words=word_list, pre=pre_list, mut=mut_list)

    belong_list = []
    data_list = []
    data_label_list = []
    mutants_list = []
    mutants_label_list = []
    str_list = []
    word_list = []
    pre_list = []
    mut_list = []
    mut_file_list = ['word_shuffling_mutants.npz', 'character_deleting_mutants.npz',
                        'symbol_insertion_mutants.npz', 'glyph_replacement_mutants.npz',
                        'character_splitting_mutants.npz', 'homophone_replacement_mutants.npz',
                        'nasal_replacement_mutants.npz', 'dorsal_replacement_mutants.npz',
                        'context_prediction_mutants.npz', 'synonym_replacement_mutants.npz',
                        'traditional_conversion_mutants.npz']
    for mut_num in range(len(mut_file_list)):
        mutant_path = os.path.join(save_path, mut_file_list[mut_num])
        temp_mutants = np.load(mutant_path, allow_pickle=True)
        belong_list.extend(temp_mutants['belong'])
        data_list.extend(temp_mutants['data'])
        data_label_list.extend(temp_mutants['data'])
        mutants_list.extend(temp_mutants['mutants'])
        mutants_label_list.extend(temp_mutants['mutants_label'])
        str_list.extend(temp_mutants['strs'])
        word_list.extend(temp_mutants['words'])
        pre_list.extend(temp_mutants['pre'])
        mut_list.extend(temp_mutants['mut'])

    if len(mutants_list)==0:
        print("一个变异成功的数据也没有！！！")
    else:
        print(len(mutants_list))
        np.savez_compressed(os.path.join(save_path, 'final_mutants'), belong=belong_list, data=data_list,
                            data_label=data_label_list, mutants=mutants_list, mutants_label=mutants_label_list,
                            strs=str_list, words=word_list, pre=pre_list, mut=mut_list)


if __name__ == '__main__':
    pass