import os

import numpy as np

def discover_poi(span1_begin, span1_end, span2_begin, span2_end, str_list, position_list, word_list):
    before_two_span = []
    between_two_span = []
    after_two_span = []
    for span_num in range(len(position_list)):
        if position_list[span_num][1] < span1_begin:
            before_two_span.append(span_num)
        elif position_list[span_num][0] >= span2_end:
            after_two_span.append(span_num)
        else:
            between_two_span.append(span_num)
    for num in range(len(before_two_span)):
        index = before_two_span[num]
        dif_value = len(word_list[index]) - len(str_list[index])
        span1_begin += dif_value
        span1_end += dif_value
        span2_begin += dif_value
        span2_end += dif_value
    for num in range(len(between_two_span)):
        index = between_two_span[num]
        dif_value = len(word_list[index]) - len(str_list[index])
        span2_begin += dif_value
        span2_end += dif_value
    return span1_begin, span1_end, span2_begin, span2_end

def get_sentence(input_x, span1_begin, span1_end, span2_begin, span2_end, str_list, position_list, word_list):

    temp_x, fina_span1_begin, final_span1_end, final_span2_begin, final_span2_end, origin_str,  replace_str = generate_sentence(input_x, span1_begin, span1_end, span2_begin, span2_end, str_list, position_list, word_list)

    return temp_x, fina_span1_begin, final_span1_end, final_span2_begin, final_span2_end, origin_str,  replace_str

def generate_sentence(input_x, span1_begin, span1_end, span2_begin, span2_end, str_list, position_list, word_list):
    origin_str = ''
    replace_str = ''
    i = 0
    temp_x = ''
    while i < len(input_x):
        for temp_num in range(len(position_list)):
            if i == position_list[temp_num][0]:
                break
            if temp_num == len(position_list) - 1:
                temp_num = len(position_list)
        if temp_num == len(position_list):
            temp_x += input_x[i]
            i += 1
        else:
            temp_x += word_list[temp_num]
            if origin_str != '':
                origin_str += '+'
                replace_str += '+'
            str = str_list[temp_num]
            origin_str += str
            temp_word = word_list[temp_num]
            replace_str += temp_word
            i = position_list[temp_num][1] + 1
    fina_span1_begin, final_span1_end, final_span2_begin, final_span2_end = discover_poi(span1_begin, span1_end, span2_begin, span2_end, str_list, position_list, word_list)
    return temp_x, fina_span1_begin, final_span1_end, final_span2_begin, final_span2_end, origin_str,  replace_str

if __name__ == '__main__':
    pass