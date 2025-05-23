import os
import math
import numpy as np

def sort_seg(data_path):
    segs = np.load(os.path.join(data_path, 'words.npy'), allow_pickle=True)
    seg_scores = np.load(os.path.join(data_path, 'scores.npy'), allow_pickle=True)
    positions = np.load(os.path.join(data_path, 'positions.npy'), allow_pickle=True)
    # print(positions)
    sort_segs = []
    sort_scores = []
    sort_positions = []
    for data_num in range(len(segs)):
        temp_list = segs[data_num]
        temp_score = seg_scores[data_num]
        temp_position = positions[data_num]
        # print(temp_list)
        # print(temp_score)
        # print(temp_position)
        # print(temp_score)
        temp_score = np.array(temp_score)
        temp_index = np.argsort(-temp_score)
        # print(temp_index)
        imp_temp_list = []
        imp_temp_score = []
        imp_temp_poi = []
        for seg_num in range(len(temp_index)):
            imp_index = temp_index[seg_num]
            imp_temp_list.append(temp_list[imp_index])
            imp_temp_score.append(temp_score[imp_index])
            imp_temp_poi.append(temp_position[imp_index])
        sort_segs.append(imp_temp_list)
        sort_scores.append(imp_temp_score)
        sort_positions.append(imp_temp_poi)
    sort_segs = np.array(sort_segs)
    sort_scores = np.array(sort_scores)
    sort_positions = np.array(sort_positions)
    np.save(os.path.join(data_path, 'sort_segs.npy'), sort_segs)
    np.save(os.path.join(data_path, 'sort_scores.npy'), sort_scores)
    np.save(os.path.join(data_path, 'sort_positions.npy'), sort_positions)
    return sort_segs, sort_scores, sort_positions

def select_words(data_path, rate):
    words_list = []
    position_list = []
    sort_segs, sort_scores, sort_positions = sort_seg(data_path)
    total_segs = sort_segs
    for data_num in range(len(sort_segs)):
        seg_num = len(total_segs[data_num])
        select_seg_num = math.ceil(rate * seg_num)
        select_words = sort_segs[data_num][:select_seg_num]
        select_words_poi = sort_positions[data_num][:select_seg_num]
        words_list.append(select_words)
        position_list.append(select_words_poi)
    words_list = np.array(words_list)
    position_list = np.array(position_list)
    save_rate_path = os.path.join(data_path, str(rate))
    if not os.path.exists(save_rate_path):
        os.makedirs(save_rate_path)
    np.save(os.path.join(save_rate_path, 'select_words.npy'), words_list)
    np.save(os.path.join(save_rate_path, 'select_words_positions.npy'), position_list)

if __name__ == '__main__':
    pass
