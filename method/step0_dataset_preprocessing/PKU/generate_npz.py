import os
import chardet
import numpy as np

def get_code(path, name):
    file_path = os.path.join(path, )
    with open('file_path', 'rb') as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
        print(encoding)

def get_sub_list(init_list, sublist_len, sep_word):
    """直接按最大长度切分"""
    list_groups = zip(*(iter(init_list),) * sublist_len)
    end_list = [list(i) + list(sep_word) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list

def getlist(input_str):
    """
    将每个输入词转换为BMES标注
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'E']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('E')
    return output_str

def get_npz(dataset_path, name):
    """
    将txt文件每一行中的文本分离出来，存储为words列表
    BMES标注法标记文本对应的标签，存储为labels
    若长度超过max_len，则直接按最大长度切分（可替换为按标点切分）
    """
    input_dir = os.path.join(dataset_path, name + '.utf8')
    output_dir = os.path.join(dataset_path, name + '.npz')
    max_len = 500
    sep_word = '@'  # 拆分句子的文本分隔符
    sep_label = 'S'  # 拆分句子的标签分隔符
    with open(input_dir, 'r', encoding='utf-8') as f:
        word_list = []
        label_list = []
        num = 0
        sep_num = 0
        for line in f:
            words = []
            line = line.strip()  # remove spaces at the beginning and the end
            if not line:
                continue  # line is None
            for i in range(len(line)):
                if line[i] == " ":
                    continue  # skip space
                words.append(line[i])
            text = line.split(" ")
            labels = []
            for item in text:
                if item == "":
                    continue
                labels.extend(getlist(item))

            if len(words) > max_len:
                # 直接按最大长度切分
                sub_word_list = get_sub_list(words, max_len - 5, sep_word)
                sub_label_list = get_sub_list(labels, max_len - 5, sep_label)
                word_list.extend(sub_word_list)
                label_list.extend(sub_label_list)
                sep_num += 1
            else:
                word_list.append(words)
                label_list.append(labels)
            num += 1
            assert len(labels) == len(words), "labels 数量与 words 不匹配"
            assert len(word_list) == len(label_list), "label 句子数量与 word 句子数量不匹配"
        print("We have", num, "lines in", name, "file processed")
        print("We have", sep_num, "lines in", name, "file get sep processed")
        # 保存成二进制文件
        np.savez_compressed(output_dir, words=word_list, labels=label_list)
        print("-------- {} data process DONE!--------".format(name))

if __name__ == "__main__":
    pass
