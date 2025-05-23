import os
import math

import numpy as np


def sort_seg(data_path):
    segs = np.load(os.path.join(data_path, 'seg.npy'), allow_pickle=True)
    seg_scores = np.load(os.path.join(data_path, 'seg_score.npy'), allow_pickle=True)
    set_texts = np.load(os.path.join(data_path, 'seg_text.npy'), allow_pickle=True)
    seg_positions = np.load(os.path.join(data_path, 'seg_position.npy'), allow_pickle=True)

    sort_segs = []
    sort_scores = []
    sort_texts = []
    sort_positions = []

    for data_num in range(len(segs)):
        temp_list = segs[data_num]
        temp_text = set_texts[data_num]
        temp_score = seg_scores[data_num]
        temp_position = seg_positions[data_num]
        # print(temp_score)
        temp_score = np.array(temp_score)
        temp_index = np.argsort(-temp_score)
        # print(temp_index)
        imp_temp_list = []
        imp_temp_score = []
        imp_temp_text = []
        imp_temp_position = []
        for seg_num in range(len(temp_index)):
            imp_index = temp_index[seg_num]
            imp_temp_list.append(temp_list[imp_index])
            imp_temp_score.append(temp_score[imp_index])
            imp_temp_text.append(temp_text[imp_index])
            imp_temp_position.append(temp_position[imp_index])
        sort_segs.append(imp_temp_list)
        sort_scores.append(imp_temp_score)
        sort_texts.append(imp_temp_text)
        sort_positions.append(imp_temp_position)

    sort_segs = np.array(sort_segs)
    sort_scores = np.array(sort_scores)
    sort_texts = np.array(sort_texts)
    sort_positions = np.array(sort_positions)
    np.save(os.path.join(data_path, 'sort_segs.npy'), sort_segs)
    np.save(os.path.join(data_path, 'sort_scores.npy'), sort_scores)
    np.save(os.path.join(data_path, 'sort_texts.npy'), sort_texts)
    np.save(os.path.join(data_path, 'sort_positions.npy'), sort_positions)
    return sort_segs, sort_scores, sort_texts, sort_positions

def select_words(data_path, rate):
    words_list = []
    word_text_list = []
    word_position_list = []
    sort_segs, sort_scores, sort_texts, sort_positions = sort_seg(data_path)
    total_segs = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)
    for data_num in range(len(sort_segs)):
        seg_num = len(total_segs[data_num][0]) + len(total_segs[data_num][1])
        select_seg_num = math.ceil(rate * seg_num)
        select_words = sort_segs[data_num][:select_seg_num]
        select_word_text = sort_texts[data_num][:select_seg_num]
        select_word_positions = sort_positions[data_num][:select_seg_num]
        words_list.append(select_words)
        word_text_list.append(select_word_text)
        word_position_list.append(select_word_positions)
    words_list = np.array(words_list)
    word_text_list = np.array(word_text_list)
    word_position_list = np.array(word_position_list)
    save_rate_path = os.path.join(data_path, str(rate))
    if not os.path.exists(save_rate_path):
        os.makedirs(save_rate_path)
    np.save(os.path.join(save_rate_path, 'select_words.npy'), words_list)
    print(words_list)
    np.save(os.path.join(save_rate_path, 'select_word_texts.npy'), word_text_list)
    print(word_text_list)
    np.save(os.path.join(save_rate_path, 'select_word_positions.npy'), word_position_list)
    print(word_position_list)


if __name__ == '__main__':
    pass
