import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
from method.step1_adv_sample_generation.PKU.create_PKU_roberta_dataset import SegDataset

fidelity_mutants_list = ['fidelity_word_shuffling_mutants.npz', 'fidelity_character_deleting_mutants.npz',
                      'fidelity_symbol_insertion_mutants.npz', 'fidelity_glyph_replacement_mutants.npz',
                      'fidelity_character_splitting_mutants.npz', 'fidelity_homophone_replacement_mutants.npz',
                      'fidelity_nasal_replacement_mutants.npz', 'fidelity_dorsal_replacement_mutants.npz',
                      'fidelity_context_prediction_mutants.npz', 'fidelity_synonym_replacement_mutants.npz',
                      'fidelity_traditional_conversion_mutants.npz']
transferability_mutants_list = ['transferability_word_shuffling_mutants.npz', 'transferability_character_deleting_mutants.npz',
                      'transferability_symbol_insertion_mutants.npz', 'transferability_glyph_replacement_mutants.npz',
                      'transferability_character_splitting_mutants.npz', 'transferability_homophone_replacement_mutants.npz',
                      'transferability_nasal_replacement_mutants.npz', 'transferability_dorsal_replacement_mutants.npz',
                      'transferability_context_prediction_mutants.npz', 'transferability_synonym_replacement_mutants.npz',
                      'transferability_traditional_conversion_mutants.npz']

def get_entities(seq):
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
    return chunks


def end_of_chunk(prev_tag, tag):
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

def evaluate_data(dev_loader, model_path):
    model = torch.load(os.path.join(model_path, 'best_PKU_roberta_base_chinese_.pt'))
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
    x_test = candidate_mutants['text']
    y_test = candidate_mutants['label']

    testing_dataset = SegDataset(x_test, y_test)
    testing_loader = DataLoader(testing_dataset, batch_size=6, collate_fn=testing_dataset.collate_fn)
    sent_data, pred_tags, true_tags, test_output = evaluate_data(testing_loader, base_model_path)
    pred_tags = np.array(pred_tags)
    return pred_tags, true_tags



def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese')
    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = np.load(os.path.join(load_path, fidelity_mutants_list[name_num]), allow_pickle=True)
        pre_labels, true_labels = cal_output_label(candidate_mutants, model_path)
        scores = []
        for data_num in range(len(pre_labels)):
            temp_score = acc_score(pre_labels[data_num], true_labels[data_num])
            scores.append(temp_score)

        belong_list = []
        ori_text_list = []
        ori_label_list = []
        text_list = []
        label_list = []
        str_list = []
        word_list = []
        mut_list = []

        for data_num in range(len(scores)):
            if scores[data_num] > 0.85:
                continue
            belong_list.append(candidate_mutants['belong'][data_num])
            ori_text_list.append(candidate_mutants['ori_text'][data_num])
            ori_label_list.append(candidate_mutants['ori_label'][data_num])
            text_list.append(candidate_mutants['text'][data_num])
            label_list.append(candidate_mutants['label'][data_num])
            str_list.append(candidate_mutants['str'][data_num])
            word_list.append(candidate_mutants['word'][data_num])
            mut_list.append(candidate_mutants['type'][data_num])

        np.savez_compressed(os.path.join(load_path, transferability_mutants_list[name_num]), belong=belong_list,
                            ori_text=ori_text_list, ori_label=ori_label_list, text=text_list, label=label_list,
                            str=str_list, word=word_list, type=mut_list)
        print(len(belong_list))

if __name__ == '__main__':
    pass
