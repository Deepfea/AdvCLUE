from hanzi_chaizi import HanziChaizi

def get_splitting_character(character):
    hc = HanziChaizi()
    try:
        characters = hc.query(character)
        result = ''
        for num in range(len(characters)):
            result += characters[num]
        return result
    except:
        return character


def get_splitting_word(temp_seg):
    final_string = ''
    i = 0
    while i < len(temp_seg):
        temp_str = temp_seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            # print(temp_str)
            temp_word = get_splitting_character(temp_str)
            final_string += temp_word
        else:
            final_string += temp_str
        i += 1
    return final_string

if __name__ == '__main__':
   pass


