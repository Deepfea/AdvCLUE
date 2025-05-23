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


def get_sentence(input_text, str_list, word_list):
    n = len(word_list)
    pair_list = get_pair(str_list, word_list, n)
    temp_sentence, temp_origin_str, temp_replace_str = generate_sentence(input_text, pair_list)

    return temp_sentence, temp_origin_str, temp_replace_str

def generate_sentence(input_text, temp_pairs):
    input_text1 = input_text
    origin_str = ''
    replace_str = ''
    for num in range(len(temp_pairs)):
        if origin_str != '':
            origin_str += '+'
            replace_str += '+'
        temp_pair = temp_pairs[num]
        # print(temp_pair)
        str = temp_pair[0]
        origin_str += str
        position = -1
        position_list = []
        while True:
            position = input_text1.find(str, position + 1)
            if position == -1:
                break
            position_list.append(position)
        temp_length = len(str)
        temp_word = temp_pair[1]
        replace_str += temp_word
        temp_sentence = ''
        text_num = 0
        while text_num < len(input_text1):
            if text_num not in position_list:
                temp_sentence += input_text1[text_num]
                text_num += 1
            else:
                temp_position = text_num
                temp_sentence += temp_word
                while text_num < temp_position + temp_length:
                    text_num += 1
        input_text1 = temp_sentence
    if input_text1 == '':
        # print(origin_str)
        input_text1 = '1'
    return input_text1, origin_str, replace_str

if __name__ == '__main__':
    pass
