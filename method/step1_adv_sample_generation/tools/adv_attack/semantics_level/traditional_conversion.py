import opencc
def get_traditional_conversion_word(seg):
    final_string = ''
    temp_string = ''
    i = 0
    while i < len(seg):
        temp_str = seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_string += temp_str
        else:
            if len(temp_string) != 0:
                temp_result = get_tradition(temp_string)
                final_string += temp_result
                temp_string = ''
            final_string += temp_str
        i += 1
    if len(temp_string) != 0:
        temp_result = get_tradition(temp_string)
        final_string += temp_result
    # print(final_string)
    return final_string

def get_tradition(seg):
    convert_to_traditional_chinese = opencc.OpenCC('s2t')
    traditional_chinese_text = convert_to_traditional_chinese.convert(seg)
    return traditional_chinese_text

if __name__ == '__main__':
    pass




