import numpy as np
import os
import pandas as pd
import jieba
from tqdm import tqdm

def context_del_answer(context_str, answer_start, answer_end):
    str1 = context_str[:answer_start]
    last_num = len(context_str) - answer_end
    str2 = context_str[-last_num:]
    # print(str1)
    # print(str2)
    return str1, str2

def discover_context_pos(temp_string, word_list, answer_start, answer_end):
    final_word_list = []
    poi = []
    num = 0
    for i in range(len(word_list)):
        temp_word = word_list[i]
        index_start = int(temp_string.find(temp_word))
        index_end = int(index_start + len(temp_word))
        if index_end <= answer_start or index_start >= answer_end:
            final_word_list.append(word_list[i])
            poi.append([])
            poi[num].append(index_start)
            poi[num].append(index_end)
            num += 1
    return final_word_list, poi

def discover_question_pos(temp_string, word_list):
    poi = []
    for i in range(len(word_list)):
        temp_word = word_list[i]
        index_start = int(temp_string.find(temp_word))
        index_end = int(index_start + len(temp_word))
        poi.append([])
        poi[i].append(index_start)
        poi[i].append(index_end)
    return poi

def del_similar_poi_word(seg_arr, position_arr):
    final_seg = []
    final_poi = []
    for data_num in tqdm(range(len(seg_arr))):
        final_seg.append([])
        final_poi.append([])
        final_data_seg = []
        final_data_poi = []
        temp_words = seg_arr[data_num][0]
        temp_positions = position_arr[data_num][0]
        for word_num in range(len(temp_words)):
            temp_word = temp_words[word_num]
            temp_position = temp_positions[word_num]
            i = 0
            num = 0
            while i < len(temp_positions):
                compare_position = temp_positions[i]
                if temp_position[0] == compare_position[0] and temp_position[1] == compare_position[1]:
                    i += 1
                    continue
                if temp_position[1] > compare_position[0] and temp_position[1] < compare_position[1]:
                    num += 1
                    break
                if temp_position[0] >= compare_position[0] and temp_position[1] <= compare_position[1]:
                    num += 1
                    break
                i += 1
            if num != 1:
                final_data_seg.append(temp_word)
                final_data_poi.append(temp_position)
        final_seg[data_num].append(final_data_seg)
        final_seg[data_num].append(seg_arr[data_num][1])
        final_poi[data_num].append(final_data_poi)
        final_poi[data_num].append(position_arr[data_num][1])
    final_seg = np.array(final_seg)
    final_poi = np.array(final_poi)
    return final_seg, final_poi

def seg_data(data_df):
    seg_list = []
    for data_num in range(len(data_df)):
        seg_list.append([])
        answer = data_df.loc[data_num, 'answer']
        answer_start = data_df.loc[data_num, 'answer_start']
        answer_end = data_df.loc[data_num, 'answer_end']
        context_segs = []
        temp_string = data_df.loc[data_num, 'context']

        str1, str2 = context_del_answer(temp_string, answer_start, answer_end)

        cut_result = jieba.cut(str1, cut_all=False)
        cut_result = list(cut_result)
        context_segs.extend(cut_result)

        cut_result = jieba.cut(str2, cut_all=False)
        cut_result = list(cut_result)
        context_segs.extend(cut_result)

        context_segs = list(set(context_segs))

        temp_string = data_df.loc[data_num, 'question']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        question_segs = list(set(cut_result))

        seg_list[data_num].append(context_segs)
        seg_list[data_num].append(question_segs)
        # break

    seg_arr1 = np.array(seg_list)
    # print(seg_arr1[0])

    seg_arr = seg_arr1
    # print(seg_arr[0])

    position_list = []
    final_seg_list = []
    for data_num in range(len(data_df)):
        position_list.append([])
        final_seg_list.append([])
        temp_string = data_df.loc[data_num, 'context']
        answer_start = data_df.loc[data_num, 'answer_start']
        answer_end = data_df.loc[data_num, 'answer_end']
        word_list = seg_arr[data_num][0]
        final_seg, context_word_poi = discover_context_pos(temp_string, word_list, answer_start, answer_end)
        position_list[data_num].append(context_word_poi)
        final_seg_list[data_num].append(final_seg)
        temp_string = data_df.loc[data_num, 'question']
        word_list = seg_arr[data_num][1]
        question_word_poi = discover_question_pos(temp_string, word_list)
        position_list[data_num].append(question_word_poi)
        final_seg_list[data_num].append(word_list)
        # break
    # print(final_seg_list[0])
    # print(position_list)
    final_seg_list, position_list = del_similar_poi_word(final_seg_list, position_list)
    # print(final_seg_list[0])
    # print(position_list)

    position_arr = np.array(position_list)
    seg_arr = np.array(final_seg_list)

    # print(seg_arr)
    return seg_arr, position_arr


if __name__ == '__main__':
    pass


