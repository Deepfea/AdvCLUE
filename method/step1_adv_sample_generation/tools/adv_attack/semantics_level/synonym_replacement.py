import numpy as np
import math
from method.step1_adv_sample_generation.tools.synonym_generation.find_synonyms import find_synonyms
import random
import time

def generate_random(number):
    current_time = int(time.time() * 1000)
    random.seed(current_time)
    random_num = random.randint(0, number-1)
    return random_num

def get_cos_value(temp_seg1, temp_seg2):
    temp_seg_list1 = list(temp_seg1)
    temp_seg_list2 = list(temp_seg2)
    temp_seg_list = list(set(temp_seg_list1 + temp_seg_list2))
    # print(temp_seg_list)
    word_fre1 = np.zeros(len(temp_seg_list), dtype='int')
    word_fre2 = np.zeros(len(temp_seg_list), dtype='int')
    for num in range(len(temp_seg_list)):
        element = temp_seg_list[num]
        word_fre1[num] = temp_seg_list1.count(element)
        word_fre2[num] = temp_seg_list2.count(element)
    # print(word_fre1)
    # print(word_fre2)
    x = 0
    y = 0
    z = 0
    for num in range(len(word_fre1)):
        x += word_fre1[num] * word_fre2[num]
        y += word_fre1[num] * word_fre1[num]
        z += word_fre2[num] * word_fre2[num]
    y = math.sqrt(y)
    z = math.sqrt(z)
    result = x / float(y * z)
    return result

def get_synonym_replacement_word(seg):
    tongyici = find_synonyms(seg)
    # print(tongyici)
    scores = []
    if len(tongyici) == 0:
        return ''
    for tongyici_num in range(len(tongyici)):
        temp_score = get_cos_value(seg, tongyici[tongyici_num])
        scores.append(temp_score)
    scores = np.array(scores)
    # print(scores)
    index_arr = np.argsort(-scores)
    temp_list = []
    temp_score = 0
    for num in range(len(index_arr)):
        if num == 0:
            temp_score = scores[index_arr[num]]
            temp_list.append(tongyici[index_arr[num]])
        elif temp_score == scores[index_arr[num]]:
            temp_list.append(tongyici[index_arr[num]])
        else:
            break
    # print(temp_list)
    result_num = len(temp_list)
    choose_num = generate_random(result_num)
    return temp_list[choose_num]

if __name__ == '__main__':
    pass

