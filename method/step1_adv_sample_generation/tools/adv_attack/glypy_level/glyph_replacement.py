import os

def initDict(path):
    dict = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            # 移除换行符，并且根据空格拆分
            splits = line.strip('\n').split(' ')
            key = splits[0]
            value = splits[1]
            dict[key] = value
    return dict


def get_glyph_word(temp_seg):
    base_path = '/media/usr/external/home/usr/project/project3_data/tools'
    bihuashuDict = initDict(os.path.join(base_path, 'bihuashu_2w.txt'))
    hanzijiegouDict = initDict(os.path.join(base_path, 'hanzijiegou_2w.txt'))
    pianpangbushouDict = initDict(os.path.join(base_path, 'pianpangbushou_2w.txt'))
    sijiaobianmaDict = initDict(os.path.join(base_path, 'sijiaobianma_2w.txt'))
    final_string = ''
    i = 0
    while i < len(temp_seg):
        temp_str = temp_seg[i]
        # print(temp_str)
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_str = get_similar_character(temp_str, hanzijiegouDict, pianpangbushouDict, bihuashuDict, sijiaobianmaDict)
        # print(temp_str)
        final_string += temp_str
        i += 1
    return final_string

def find_all_occurrences(lst, value, characters):
    indices = [index for index, item in enumerate(lst) if item == value]
    character_list = []
    for i in range(len(indices)):
        character_list.append(characters[indices[i]])
    return character_list


def get_similar_character(temp_str, hanzijiegouDict, pianpangbushouDict, bihuashuDict, sijiaobianmaDict):
    pianpangbushou_key_list = list(pianpangbushouDict.keys())
    pianpangbushou_val_list = list(pianpangbushouDict.values())
    try:
        ind = pianpangbushou_key_list.index(temp_str)
        # print(ind)
    except ValueError:
        return temp_str
    pianpangbushou = pianpangbushou_val_list[ind]
    pianpangbushou_characters = find_all_occurrences(pianpangbushou_val_list, pianpangbushou, pianpangbushou_key_list)
    # print(pianpangbushou_characters)

    hanzijiegou_key_list = list(hanzijiegouDict.keys())
    hanzijiegou_val_list = list(hanzijiegouDict.values())
    try:
        ind = hanzijiegou_key_list.index(temp_str)
    except ValueError:
        return temp_str
    hanzijiegou = hanzijiegou_val_list[ind]
    hanzijiegou_characters = find_all_occurrences(hanzijiegou_val_list, hanzijiegou, hanzijiegou_key_list)
    # print(hanzijiegou_characters)

    # sijiaobianma_key_list = list(sijiaobianmaDict.keys())
    # sijiaobianma_val_list = list(sijiaobianmaDict.values())
    # try:
    #     ind = sijiaobianma_key_list.index(temp_str)
    # except ValueError:
    #     return temp_str
    # sijiaobianma = sijiaobianma_val_list[ind]
    # sijiaobianma_characters = find_all_occurrences(sijiaobianma_val_list, sijiaobianma, sijiaobianma_key_list)
    # print(sijiaobianma_characters)

    # 将两个列表转换为集合
    set1 = set(pianpangbushou_characters)
    set2 = set(hanzijiegou_characters)
    # set3 = set(sijiaobianma_characters)
    # 使用集合的交集操作找出重复元素
    # duplicates = set1 | set2 |set3
    duplicates = set1 & set2
    duplicates = set(duplicates)
    duplicates = list(duplicates)
    # print(duplicates)
    if len(duplicates) == 1:
        return temp_str
    max_value = 100
    final_str = ''
    for i in range(len(duplicates)):
        if duplicates[i] == temp_str:
            continue
        try:
            x = int(bihuashuDict[temp_str])
        except KeyError:
            x = 1000
        try:
            y = int(bihuashuDict[duplicates[i]])
        except KeyError:
            y = -1000
        temp_value = abs(x - y)
        if temp_value == 0:
            final_str = duplicates[i]
            break
        if temp_value < max_value:
            max_value = temp_value
            final_str = duplicates[i]
    return final_str


if __name__ == '__main__':
    pass