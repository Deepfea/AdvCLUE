
flag_list = ['。', '.', '，', ',', '；', ';', '？', '?',
        '、', '！', '!', '“', '”', '"', '《', '》',
        '~', '<', '>', '：', ':', '%', '/',
        '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
def get_word_label(word):
    label = []
    if len(word) == 0:
        label = []
    elif len(word) == 1:
        label.append('S')
    else:
        string = ''
        i = 0
        while i < len(word):
            if word[i] not in flag_list:
                string += word[i]
                i += 1
            else:
                if len(string) == 0:
                    label.append('S')
                    i += 1
                elif len(string) == 1:
                    label.append('S')
                    label.append('S')
                    string = ''
                    i += 1
                else:
                    label.append('B')
                    temp_num = len(string) - 2
                    for temp_i in range(temp_num):
                        label.append('M')
                    label.append('E')
                    label.append('S')
                    i += 1
                    string = ''
        if len(string) == 1:
            label.append('S')
        elif len(string) >= 2:
            label.append('B')
            temp_num = len(string) - 2
            for temp_i in range(temp_num):
                label.append('M')
            label.append('E')
    return label


def get_word_labels(word_list):
    word_label_list = []
    for word_num in range(len(word_list)):
        word = word_list[word_num]
        word_label = get_word_label(word)
        change_word = list(word)
        word_label_list.append(word_label)
    return word_label_list

def get_sentence(input_x, input_y, str_list, position_list, word_list):
    word_label_list = get_word_labels(word_list)
    temp_x, temp_y, origin_str,  replace_str = generate_sentence(input_x, input_y, str_list, position_list, word_list, word_label_list)

    return temp_x, temp_y, origin_str,  replace_str

def generate_sentence(input_x, input_y, str_list, position_list, word_list, word_label_list):
    origin_str = ''
    replace_str = ''
    i = 0
    temp_x = []
    temp_y = []
    while i < len(input_x):
        for temp_num in range(len(position_list)):
            if i == position_list[temp_num][0]:
                break
            if temp_num == len(position_list) - 1:
                temp_num = len(position_list)
        if temp_num == len(position_list):
            temp_x.append(input_x[i])
            temp_y.append(input_y[i])
            i += 1
        else:
            temp_x.extend(list(word_list[temp_num]))
            temp_y.extend(word_label_list[temp_num])
            if origin_str != '':
                origin_str += '+'
                replace_str += '+'
            str = str_list[temp_num]
            origin_str += str
            temp_word = word_list[temp_num]
            replace_str += temp_word
            i = position_list[temp_num][1] + 1
        # print(temp_x)
        # print(temp_y)
        # print(origin_str)
        # print(replace_str)
        # print(i)
    return temp_x, temp_y, origin_str,  replace_str

if __name__ == '__main__':
    pass