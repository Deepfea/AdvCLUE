import random
import time


def generate_random(temp_seg, seg_len):

    if seg_len == 2:
        temp_string = ''
        temp_string += temp_seg[1]
        temp_string += temp_seg[0]
    else:
        temp_seg = list(temp_seg)
        current_time = int(time.time() * 1000)
        random.seed(current_time)
        random.shuffle(temp_seg)
        temp_string = ''
        for num in range(seg_len):
            temp_string += temp_seg[num]
    return temp_string


def get_shuffle_word(temp_seg):
    seg_len = len(temp_seg)
    if seg_len == 1:
        temp_string = temp_seg
    else:
        temp_string = generate_random(temp_seg, seg_len)
    return temp_string

if __name__ == '__main__':
    pass



