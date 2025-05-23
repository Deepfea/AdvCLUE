import itertools
import random

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


def get_sentence(text_a, text_b, str_list, str_text_list, word_list):
    temp_sentence_a, temp_sentence_b, temp_origin_str, temp_replace_str = generate_sentence(text_a, text_b, str_list, str_text_list, word_list)

    return temp_sentence_a, temp_sentence_b, temp_origin_str, temp_replace_str

def generate_sentence(text_a, text_b, str_list, str_text_list, word_list):
    origin_str = ''
    replace_str = ''
    for num in range(len(str_list)):
        if origin_str != '':
            origin_str += '+'
            replace_str += '+'
        str = str_list[num]
        origin_str += str
        temp_word = word_list[num]
        replace_str += temp_word
        if str_text_list[num] == 'a':
            text_a = text_a.replace(str, temp_word)
        elif str_text_list[num] == 'b':
            text_b = text_b.replace(str, temp_word)
    if text_a == '':
        # print(origin_str)
        text_a = '1'
    if text_b == '':
        text_b = '1'
    return text_a, text_b, origin_str, replace_str

if __name__ == '__main__':
    pass