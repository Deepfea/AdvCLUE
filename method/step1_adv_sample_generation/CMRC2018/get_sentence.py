import itertools
import random

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
import os

def get_pair(str_list, word_list, n):
    pair_list = []
    for i in range(n):
        temp_list = []
        temp_list.append(str_list[i])
        temp_list.append(word_list[i])
        pair_list.append(temp_list)
    return pair_list


def get_sentence(text_a, text_b, answer_start, answer_end, str_list, str_text_list, str_position_list, word_list):
    temp_sentence_a, temp_sentence_b, temp_answer_start, temp_answer_end, temp_origin_str, temp_replace_str, temp_str_text = \
        generate_sentence(text_a, text_b, answer_start, answer_end, str_list, str_text_list, str_position_list, word_list)

    return temp_sentence_a, temp_sentence_b, temp_answer_start, temp_answer_end, temp_origin_str, temp_replace_str, temp_str_text


def get_a_and_b_str(str_list, str_text_list, str_position_list, word_list):
    str_a_list = []
    str_b_list = []
    str_text_a_list = []
    str_text_b_list = []
    str_position_a_list = []
    str_position_b_list = []
    word_a_list = []
    word_b_list = []
    for num in range(len(str_list)):
        if str_text_list[num] == 'a':
            str_a_list.append(str_list[num])
            str_text_a_list.append(str_text_list[num])
            str_position_a_list.append(str_position_list[num])
            word_a_list.append(word_list[num])
        else:
            str_b_list.append(str_list[num])
            str_text_b_list.append(str_text_list[num])
            str_position_b_list.append(str_position_list[num])
            word_b_list.append(word_list[num])
    return str_a_list, str_b_list, str_text_a_list, str_text_b_list, str_position_a_list, str_position_b_list, word_a_list, word_b_list

def discover_answer_start_and_end(answer_start, answer_end, str_a_list, str_position_a_list, word_a_list):
    value = 0
    for num in range(len(str_a_list)):
        if str_position_a_list[num][0] >= answer_end:
            continue
        difference = len(word_a_list[num]) - len(str_a_list[num])
        value += difference
    return answer_start + value, answer_end + value

def get_new_word_list(word_list):
    final_word_list = []
    for num in range(len(word_list)):
        temp_word = word_list[num]
        if '\uf7ee' in temp_word:
            temp_word = temp_word.replace('\uf7ee', '')
        final_word_list.append(temp_word)
    return final_word_list

def poi_str_to_list(str_position_list):
    final_position_list = []
    for num in range(len(str_position_list)):
        temp_list = []
        temp_str = str_position_list[num]
        temp_str = temp_str.replace('[', '')
        temp_str = temp_str.replace(']', '')
        temp_str = temp_str.replace(' ', '')
        temp_str = temp_str.split(',')
        start_index = int(temp_str[0])
        end_index = int(temp_str[1])
        temp_list.append(start_index)
        temp_list.append(end_index)
        final_position_list.append(temp_list)
    return final_position_list

def generate_sentence(text_a, text_b, answer_start, answer_end, str_list, str_text_list, str_position_list, word_list):
    word_list = get_new_word_list(word_list)
    str_position_list = poi_str_to_list(str_position_list)
    origin_str = ''
    replace_str = ''
    str_text = ''

    final_text_a = ''
    final_text_b = ''


    str_a_list, str_b_list, str_text_a_list, str_text_b_list, str_position_a_list, str_position_b_list, word_a_list, word_b_list = get_a_and_b_str(str_list, str_text_list, str_position_list, word_list)
    print(str_a_list)
    print(str_text_a_list)
    print(str_position_a_list)
    print(word_a_list)
    i = 0
    while i < len(text_a):
        if len(str_position_a_list) == 0:
            final_text_a = text_a
            break
        for temp_num in range(len(str_position_a_list)):
            if i == str_position_a_list[temp_num][0]:
                break
            if temp_num == len(str_position_a_list) - 1:
                temp_num = len(str_position_a_list)
        if temp_num == len(str_position_a_list):
            final_text_a += text_a[i]
            i += 1
        else:
            final_text_a += word_a_list[temp_num]
            if origin_str != '':
                origin_str += '+'
                replace_str += '+'
                str_text += '+'
            str = str_a_list[temp_num]
            origin_str += str
            temp_word = word_a_list[temp_num]
            replace_str += temp_word
            str_text += 'a'
            i = str_position_a_list[temp_num][1]
    i = 0
    while i < len(text_b):
        if len(str_position_b_list) == 0:
            final_text_b = text_b
            break
        for temp_num in range(len(str_position_b_list)):
            if i == str_position_b_list[temp_num][0]:
                break
            if temp_num == len(str_position_b_list) - 1:
                temp_num = len(str_position_b_list)
        if temp_num == len(str_position_b_list):
            final_text_b += text_b[i]
            i += 1
        else:
            final_text_b += word_b_list[temp_num]
            if origin_str != '':
                origin_str += '+'
                replace_str += '+'
                str_text += '+'
            str = str_b_list[temp_num]
            origin_str += str
            temp_word = word_b_list[temp_num]
            replace_str += temp_word
            str_text += 'b'
            i = str_position_b_list[temp_num][1]

    if final_text_a == '':
        # print(origin_str)
        final_text_a = '1'
    if final_text_b == '':
        final_text_b = '1'
    answer_start, answer_end = discover_answer_start_and_end(answer_start, answer_end, str_a_list, str_position_a_list, word_a_list)

    return final_text_a, final_text_b, answer_start, answer_end, origin_str, replace_str, str_text

if __name__ == '__main__':
    pass



