import random
import time
symbol_list = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
symbol_num = len(symbol_list)

def get_symbol_insert_word(temp_seg):
    temp_seg = list(temp_seg)
    seg_num = len(temp_seg)
    current_time = int(time.time() * 1000)
    random.seed(current_time)
    random_int1 = random.randint(0, seg_num-1)
    current_time = int(time.time() * 1000)
    random.seed(current_time)
    random_int2 = random.randint(0, symbol_num-1)
    # print(random_int1)
    # print(random_int2)
    temp_string = ''
    for num in range(seg_num):
        if num == random_int1:
            temp_string += symbol_list[random_int2]
        temp_string += temp_seg[num]
    return temp_string

if __name__ == '__main__':
    pass


