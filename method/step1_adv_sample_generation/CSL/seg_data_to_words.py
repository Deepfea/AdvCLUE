import numpy as np
import os
import pandas as pd
import jieba

def get_keyword_word(keyword):
    word_list = []
    string = ''
    for num in range(len(keyword)):
        if keyword[num] == ';':
            word_list.append(string)
            string = ''
        else:
            string += keyword[num]
    word_list = list(set(word_list))
    return word_list

def del_keywords(temp_string, str_list):
    keyword_num = 0
    temp_string_list = []
    temp_string_list.append(temp_string)

    while keyword_num < len(str_list):
        temp_string = []
        for temp_string_num in range(len(temp_string_list)):
            temp_text = temp_string_list[temp_string_num]
            temp_texts = temp_text.split(str_list[keyword_num])
            temp_string.extend(temp_texts)
        temp_string_list = temp_string
        keyword_num += 1
    final_string_list = []
    for num in range(len(temp_string_list)):
        if len(temp_string_list[num]) == 0:
            continue
        final_string_list.append(temp_string_list[num])
    return final_string_list

def get_abs_word(abs_list):
    word_list = []
    for num in range(len(abs_list)):
        cut_result = jieba.cut(abs_list[num], cut_all=False)
        cut_result = list(cut_result)
        word_list.extend(cut_result)
    word_list = list(set(word_list))
    return word_list

def seg_data(data_df):

    seg_list = []
    for data_num in range(len(data_df)):
        segs = []

        temp_string = data_df.loc[data_num, 'keyword']
        keyword_word_list = get_keyword_word(temp_string)

        temp_string = data_df.loc[data_num, 'abs']

        abs_list = del_keywords(temp_string, keyword_word_list)
        abs_word_list = get_abs_word(abs_list)

        segs.append(abs_word_list)
        segs.append(keyword_word_list)

        seg_list.append(segs)

    seg_arr = np.array(seg_list)
    return seg_arr


if __name__ == '__main__':

    pass


