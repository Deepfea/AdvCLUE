from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultDagParams, dag, simplify_pinyin

def simplify_py(pinyin_list):
    simplify_pinyin_list = []
    for i in range(len(pinyin_list)):
        temp = pinyin_list[i]
        temp = simplify_pinyin(temp)
        simplify_pinyin_list.append(temp)
    return simplify_pinyin_list

def pinyin_2_hanzi(pinyin_list, str, max_num = 10):
    pinyin_list = simplify_py(pinyin_list)
    # print(pinyin_list)
    dagParams = DefaultDagParams()
    result = dag(dagParams, pinyin_list, path_num=max_num, log=True)
    # print(result)
    if len(result) == 0:
        return str
    if len(result) < max_num:
        max_num = len(result)
    for i in range(max_num):
        temp_string = ''
        string_list = result[i].path
        for num in range(len(string_list)):
            temp_string += string_list[num]
        if temp_string != str:
            break
    return temp_string

def nasal_replacement(temp_pinyin):
    nasal1 = ['ang', 'an', 'eng', 'en', 'ing', 'in']
    nasal2 = ['an', 'ang', 'en', 'eng', 'in', 'ing']
    for num in range(len(temp_pinyin)):
        for nasal_num in range(len(nasal1)):
            if nasal1[nasal_num] in temp_pinyin[num]:
                temp_pinyin[num] = temp_pinyin[num].replace(nasal1[nasal_num], nasal2[nasal_num])
                break
    return temp_pinyin

def get_nasal_replacement_word(seg):
    # print(seg)
    final_string = ''
    temp_string = ''
    i = 0
    while i < len(seg):
        temp_str = seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_string += temp_str
        else:
            if len(temp_string) != 0:
                temp_pinyin = lazy_pinyin(temp_string)
                # print(temp_pinyin)
                temp_pinyin = nasal_replacement(temp_pinyin)
                # print(temp_pinyin)
                temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
                final_string += temp_result
                temp_string = ''
            final_string += temp_str
        i += 1
    if len(temp_string) != 0:
        temp_pinyin = lazy_pinyin(temp_string)
        # print(temp_pinyin)
        temp_pinyin = nasal_replacement(temp_pinyin)
        # print(temp_pinyin)
        temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
        final_string += temp_result
    # print(final_string)
    return final_string

if __name__ == '__main__':
    pass





