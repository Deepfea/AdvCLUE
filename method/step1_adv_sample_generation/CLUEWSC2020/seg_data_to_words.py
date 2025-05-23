import numpy as np
import os
import pandas as pd
import jieba

remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
def get_words(temp_string, cut_result, temp_span1_begin, temp_span1_end, temp_span2_begin, temp_span2_end):
    words = []
    position = []
    total_words_num = 0
    for cut_num in range(len(cut_result)):
        temp_words = cut_result[cut_num]
        if temp_words in remove_flag:
            total_words_num += 1
            continue
        temp_poi = []
        position_start = temp_string.find(temp_words)
        position_end = position_start + len(temp_words) - 1
        if (position_start >= temp_span1_begin and position_start <= temp_span1_end-1) or (position_end >= temp_span1_begin and position_end <= temp_span1_end-1):
            continue
        if (position_start >= temp_span2_begin and position_start <= temp_span2_end-1) or (position_end >= temp_span2_begin and position_end <= temp_span2_end-1):
            continue
        else:
            words.append(temp_words)
            temp_poi.append(position_start)
            temp_poi.append(position_end)
            position.append(temp_poi)
            total_words_num += 1
    return words, position, total_words_num + 2

def seg_data(data_df):
    seg_list = []
    poi_list = []
    words_num_list = []
    for data_num in range(len(data_df)):
        temp_string = data_df.loc[data_num, 'text']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        cut_result = list(set(cut_result))
        temp_span1_begin = data_df.loc[data_num, 'span1_begin']
        temp_span1_end = data_df.loc[data_num, 'span1_end']
        temp_span2_begin = data_df.loc[data_num, 'span2_begin']
        temp_span2_end = data_df.loc[data_num, 'span2_end']
        temp_seg_list, temp_poi_list, words_num = get_words(temp_string, cut_result, temp_span1_begin, temp_span1_end, temp_span2_begin, temp_span2_end)
        seg_list.append(temp_seg_list)
        poi_list.append(temp_poi_list)
        words_num_list.append(words_num)
    seg_arr = np.array(seg_list)
    poi_arr = np.array(poi_list)
    words_num_arr = np.array(words_num_list)
    return seg_arr, poi_arr, words_num_arr


if __name__ == '__main__':
    pass

