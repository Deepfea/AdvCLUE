import random
import time

import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
import os

model_name = 'bert_base_chinese_mlm'
load_path = os.path.join('/media/usr/external/home/usr/project/project3_data/base_model', model_name)
tokenizer = AutoTokenizer.from_pretrained(load_path)
model = AutoModelForMaskedLM.from_pretrained(load_path)

def remove_element(lst, element):
    return [x for x in lst if x != element]

def random_select_word(word_list, n):
    final_list = []
    if len(word_list) < n:
        final_list = word_list
    else:
        index = random.sample(range(0, len(word_list)), n)
        for index_num in range(len(index)):
            final_list.append(word_list[index[index_num]])
    return final_list

def add_mask(input, str):
    temp_position = input.find(str, 0)
    temp_sentence = ''
    temp_length = len(str)
    text_num = 0
    for text_num in range(len(input)):
        if text_num == temp_position:
            temp_sentence += '[MASK]'
        elif (text_num > temp_position) and (text_num < temp_position + temp_length):
            continue
        else:
            temp_sentence += input[text_num]
    return temp_sentence

def find_position(input_list, str='[MASK]'):
    position = -1
    if len(input_list) > 512:
        input_list = input_list[:512]
    # print(input_list)
    for input_list_num in range(len(input_list)):
        temp = input_list[input_list_num]
        if temp == str:
            position = input_list_num
            break
    return input_list, position

def find_position1(input_list, str='[MASK]'):
    position = -1
    if len(input_list) > 511:
        input_list = input_list[:511]
    # print(input_list)
    for input_list_num in range(len(input_list)):
        temp = input_list[input_list_num]
        if temp == str:
            position = input_list_num
            break
    return input_list, position
def remove_flag(word_list):
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    for remove_flag_num in range(len(remove_flag)):
        temp_flag = remove_flag[remove_flag_num]
        if temp_flag in word_list:
            word_list = remove_element(word_list, temp_flag)
    return word_list

def get_n_1_word(input, str, n1=20):
    word_list = []
    temp_sentence = add_mask(input, str)
    # print(temp_sentence)
    temp_tokenized_text = tokenizer.tokenize(temp_sentence)
    temp_tokenized_text, temp_position = find_position(temp_tokenized_text)
    if temp_position == -1:
        return word_list
    # print("加入第一个[MASK]后，句子为：")
    # print(temp_tokenized_text)
    masked_input = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
    segment_ids = [0] * len(masked_input)

    masked_input = torch.tensor([masked_input])
    segment_ids = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = model(masked_input, token_type_ids=segment_ids)
        predictions = outputs[0]
    arr = predictions[0, temp_position].numpy()
    sorted_indices = np.argsort(arr)[::-1]
    predict_list = []
    for i in range(n1):
        predict_list.append(sorted_indices[i])
    masked_predictions = tokenizer.convert_ids_to_tokens(predict_list)
    # print(masked_predictions)
    word_list = remove_flag(masked_predictions)
    # print(word_list)
    return word_list


def get_n_2_word(input, str, n1=15, n2=2):
    word_list = []
    temp_sentence = add_mask(input, str)
    temp_tokenized_text = tokenizer.tokenize(temp_sentence)
    temp_tokenized_text, temp_position = find_position1(temp_tokenized_text)
    if temp_position == -1:
        return word_list
    # print("加入第一个[MASK]后，句子为：")
    # print(temp_tokenized_text)

    masked_input = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
    # print(masked_input)
    segment_ids = [0] * len(masked_input)
    masked_input = torch.tensor([masked_input])
    segment_ids = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = model(masked_input, token_type_ids=segment_ids)
        predictions = outputs[0]
    arr = predictions[0, temp_position].numpy()
    sorted_indices = np.argsort(arr)[::-1]
    predict_list = []
    for i in range(n1):
        predict_list.append(sorted_indices[i])
    word_list1 = tokenizer.convert_ids_to_tokens(predict_list)
    word_list1 = remove_flag(word_list1)
    # print(word_list1)
    for word_list_num in range(len(word_list1)):
        word1 = word_list1[word_list_num]
        temp_tokenized_text1 = []
        for text_num in range(len(temp_tokenized_text)):
            if text_num == temp_position:
                temp_tokenized_text1.append(word1)
            temp_tokenized_text1.append(temp_tokenized_text[text_num])
        # print("加入第二个[MASK]后，句子为：")
        # print(temp_tokenized_text1)
        masked_input = tokenizer.convert_tokens_to_ids(temp_tokenized_text1)
        segment_ids = [0] * len(masked_input)
        masked_input = torch.tensor([masked_input])
        segment_ids = torch.tensor([segment_ids])
        with torch.no_grad():
            outputs = model(masked_input, token_type_ids=segment_ids)
            predictions = outputs[0]
        arr = predictions[0, temp_position+1].numpy()
        sorted_indices = np.argsort(arr)[::-1]
        predict_list = []
        for i in range(n2):
            predict_list.append(sorted_indices[i])
        word_list2 = tokenizer.convert_ids_to_tokens(predict_list)
        word_list2 = remove_flag(word_list2)
        # print(word_list2)
        for word_list2_num in range(len(word_list2)):
            word2 = word_list2[word_list2_num]
            word = word1 + word2
            word_list.append(word)
    # print(word_list)
    return word_list

def generate_sentence(input_text, str, word_list):
    position = -1
    position_list = []
    while True:
        position = input_text.find(str, position + 1)
        if position == -1:
            break
        position_list.append(position)
    temp_length = len(str)
    sentence_list = []
    for word_num in range(len(word_list)):
        temp_word = word_list[word_num]
        temp_sentence = ''
        text_num = 0
        while text_num < len(input_text):
            if text_num not in position_list:
                temp_sentence += input_text[text_num]
                text_num += 1
            else:
                temp_position = text_num
                temp_sentence += temp_word
                while text_num < temp_position + temp_length:
                    text_num += 1
        sentence_list.append(temp_sentence)
    # print(sentence_list)
    return sentence_list

def generate_random(number):
    current_time = int(time.time() * 1000)
    random.seed(current_time)
    random_num = random.randint(0, number-1)
    return random_num

def get_context_prediction_word(fact, temp_seg):
    word1_list = get_n_1_word(fact, temp_seg, n1=2)
    word2_list = get_n_2_word(fact, temp_seg, n1=2, n2=2)
    word_list = word1_list + word2_list
    if len(word_list) == 0:
        word = ''
    else:
        word = word_list[generate_random(len(word_list))]
    return word

if __name__ == '__main__':
    pass


