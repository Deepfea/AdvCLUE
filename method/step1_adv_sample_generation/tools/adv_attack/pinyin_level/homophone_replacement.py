import random
import time
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultDagParams, dag, simplify_pinyin
import torch
import numpy as np
import os

def simplify_py(pinyin_list):
    simplify_pinyin_list = []
    for i in range(len(pinyin_list)):
        temp = pinyin_list[i]
        temp = simplify_pinyin(temp)
        simplify_pinyin_list.append(temp)
    return simplify_pinyin_list

def pinyin_2_hanzi(pinyin_list, str, max_num = 10):
    pinyin_list = simplify_py(pinyin_list)
    # print(pinyin_list)
    dagParams = DefaultDagParams()
    result = dag(dagParams, pinyin_list, path_num=max_num, log=True)
    # print(result)
    if len(result) == 0:
        return str
    if len(result) < max_num:
        max_num = len(result)
    for i in range(max_num):
        temp_string = ''
        string_list = result[i].path
        for num in range(len(string_list)):
            temp_string += string_list[num]
        if temp_string != str:
            break
    return temp_string

def get_homophone_replacement_word(seg):
    # print(seg)
    final_string = ''
    temp_string = ''
    i = 0
    while i < len(seg):
        temp_str = seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_string += temp_str
        else:
            if len(temp_string) != 0:
                temp_pinyin = lazy_pinyin(temp_string)
                temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
                final_string += temp_result
                temp_string = ''
            final_string += temp_str
        i += 1
    if len(temp_string) != 0:
        temp_pinyin = lazy_pinyin(temp_string)
        temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
        final_string += temp_result
    # print(final_string)
    return final_string

if __name__ == '__main__':
    pass



