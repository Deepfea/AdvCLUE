import random
import time

def get_character_deleting_word(temp_seg):
    temp_seg = list(temp_seg)
    seg_num = len(temp_seg)
    current_time = int(time.time() * 1000)
    random.seed(current_time)
    random_int = random.randint(0, seg_num-1)
    temp_string = ''
    for num in range(seg_num):
        if num == random_int:
            continue
        temp_string += temp_seg[num]
    return temp_string

if __name__ == '__main__':
    pass



