import numpy as np
import os
from tqdm import tqdm
import pandas as pd

from method.step1_adv_sample_generation.tools.adv_attack.glypy_level.glyph_replacement import get_glyph_word
from method.step1_adv_sample_generation.tools.adv_attack.glypy_level.character_splitting import get_splitting_word
from method.step1_adv_sample_generation.tools.adv_attack.basis_level.word_shuffling import get_shuffle_word
from method.step1_adv_sample_generation.tools.adv_attack.basis_level.symbol_insertion import get_symbol_insert_word
from method.step1_adv_sample_generation.tools.adv_attack.basis_level.character_deleting import get_character_deleting_word
from method.step1_adv_sample_generation.tools.adv_attack.pinyin_level.dorsal_word_replacement import get_dorsal_replacement_word
from method.step1_adv_sample_generation.tools.adv_attack.pinyin_level.nasal_word_replacement import get_nasal_replacement_word
from method.step1_adv_sample_generation.tools.adv_attack.pinyin_level.homophone_replacement import get_homophone_replacement_word
from method.step1_adv_sample_generation.tools.adv_attack.semantics_level.synonym_replacement import get_synonym_replacement_word
from method.step1_adv_sample_generation.tools.adv_attack.semantics_level.traditional_conversion import get_traditional_conversion_word
from method.step1_adv_sample_generation.tools.adv_attack.semantics_level.context_prediction import get_context_prediction_word
from method.step1_adv_sample_generation.CLUEWSC2020.get_sentence import get_sentence

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

def del_str(x, y):
    del_string = '\uf7ee'
    temp_x = []
    temp_y = []
    for num in range(len(x)):
        if x[num] == del_string:
            continue
        temp_x.append(x[num])
        temp_y.append(y[num])
    return temp_x, temp_y

def save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list,
                           span2_begin_list, span2_end_list, ori_strs, rep_strs , load_rate_path, save_name):

    output_dir = os.path.join(load_rate_path, save_name)
    fact_list = []
    for data_num in range(len(test_data)):
        fact = test_data.loc[data_num, 'text']
        fact_list.append(fact)
    merge_dt_dict = {'fact': fact_list, 'text': sentences, 'str': ori_strs, 'word': rep_strs, 'label_id': label_list,
                     'span1_begin': span1_begin_list, 'span1_end': span1_end_list,
                     'span2_begin': span2_begin_list, 'span2_end': span2_end_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(output_dir, index=False)

def get_sentences(test_data, select_words, select_words_positions, mutate_list):
    sentences = []
    span1_begin_list = []
    span1_end_list = []
    span2_begin_list = []
    span2_end_list = []
    label_list = []
    ori_strs = []
    rep_strs = []

    for data_num in range(len(select_words)):
        temp_x = test_data.loc[data_num, 'text']
        temp_y = test_data.loc[data_num, 'label_id']
        words = select_words[data_num]
        words_position = select_words_positions[data_num]
        span1_begin = test_data.loc[data_num, 'span1_begin']
        span1_end = test_data.loc[data_num, 'span1_end']
        span2_begin = test_data.loc[data_num, 'span2_begin']
        span2_end = test_data.loc[data_num, 'span2_end']
        temp_mutate_list = mutate_list[data_num]
        x, fina_span1_begin, final_span1_end, final_span2_begin, final_span2_end, origin_str,  replace_str = get_sentence(temp_x, span1_begin, span1_end, span2_begin, span2_end, words, words_position, temp_mutate_list)
        sentences.append(x)
        label_list.append(temp_y)
        span1_begin_list.append(fina_span1_begin)
        span1_end_list.append(final_span1_end)
        span2_begin_list.append(final_span2_begin)
        span2_end_list.append(final_span2_end)
        ori_strs.append(origin_str)
        rep_strs.append(replace_str)
    return sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs

def get_fact(x_test):
    fact_list = []
    for num in range(len(x_test)):
        temp_fact = ''
        for seg_num in range(len(x_test[num])):
            temp_fact += x_test[num][seg_num]
        fact_list.append(temp_fact)
    return fact_list

def generate_mutant(data_path, rate):
    print(rate)
    load_rate_path = os.path.join(data_path, str(rate))
    select_words = np.load(os.path.join(load_rate_path, 'select_words.npy'), allow_pickle=True)
    select_words_positions = np.load(os.path.join(load_rate_path, 'select_word_positions.npy'), allow_pickle=True)

    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

    character_deleting_list = []
    symbol_insertion_list = []
    word_shuffling_list = []
    character_splitting_list = []
    glyph_replacement_list = []
    dorsal_replacement_list = []
    homophone_replacement_list = []
    nasal_replacement_list = []
    context_prediction_list = []
    synonym_replacement_list = []
    traditional_conversion_list = []

    for data_num in tqdm(range(len(test_data))):
        seg_list = list(select_words[data_num])
        fact = test_data.loc[data_num, 'text']
        data_character_deleting_list = []
        data_symbol_insertion_list = []
        data_word_shuffling_list = []
        data_character_splitting_list = []
        data_glyph_replacement_list = []
        data_dorsal_replacement_list = []
        data_homophone_replacement_list = []
        data_nasal_replacement_list = []
        data_context_prediction_list = []
        data_synonym_replacement_list = []
        data_traditional_conversion_list = []
        for seg_num in range(len(seg_list)):
            temp_seg = seg_list[seg_num]

            # 打乱词语中的字
            shuffle_seg = get_shuffle_word(temp_seg)
            data_word_shuffling_list.append(shuffle_seg)

            # 词语中插入符号
            symbol_seg = get_symbol_insert_word(temp_seg)
            data_symbol_insertion_list.append(symbol_seg)

            #删除某个字符
            delete_seg = get_character_deleting_word(temp_seg)
            data_character_deleting_list.append(delete_seg)

            # 字形攻击
            glyph_seg = get_glyph_word(temp_seg)
            data_glyph_replacement_list.append(glyph_seg)

            # 分割词语中的字
            splitting_seg = get_splitting_word(temp_seg)
            data_character_splitting_list.append(splitting_seg)

            #舌头发音
            dorsal_seg = get_dorsal_replacement_word(temp_seg)
            data_dorsal_replacement_list.append(dorsal_seg)

            #鼻子发音
            nasal_seg = get_nasal_replacement_word(temp_seg)
            data_nasal_replacement_list.append(nasal_seg)

            #通拼音词
            homophone_seg = get_homophone_replacement_word(temp_seg)
            data_homophone_replacement_list.append(homophone_seg)

            #同义词
            synonym_seg = get_synonym_replacement_word(temp_seg)
            data_synonym_replacement_list.append(synonym_seg)

            #上下文预测
            context_seg = get_context_prediction_word(fact, temp_seg)
            data_context_prediction_list.append(context_seg)

            #简繁转换
            traditional_seg = get_traditional_conversion_word(temp_seg)
            data_traditional_conversion_list.append(traditional_seg)

        word_shuffling_list.append(data_word_shuffling_list)
        character_deleting_list.append(data_character_deleting_list)
        symbol_insertion_list.append(data_symbol_insertion_list)
        glyph_replacement_list.append(data_glyph_replacement_list)
        character_splitting_list.append(data_character_splitting_list)
        homophone_replacement_list.append(data_homophone_replacement_list)
        nasal_replacement_list.append(data_nasal_replacement_list)
        dorsal_replacement_list.append(data_dorsal_replacement_list)
        context_prediction_list.append(data_context_prediction_list)
        synonym_replacement_list.append(data_synonym_replacement_list)
        traditional_conversion_list.append(data_traditional_conversion_list)
    save_word_list = ['word_shuffling_candidate_words.npy', 'character_deleting_candidate_words.npy',
                      'symbol_insertion_candidate_words.npy', 'glyph_replacement_candidate_words.npy',
                      'character_splitting_candidate_words.npy', 'homophone_replacement_candidate_words.npy',
                      'nasal_replacement_candidate_words.npy', 'dorsal_replacement_candidate_words.npy',
                      'context_prediction_candidate_words.npy', 'synonym_replacement_candidate_words.npy',
                      'traditional_conversion_candidate_words.npy']
    np.save(os.path.join(load_rate_path, save_word_list[0]), np.array(word_shuffling_list))
    np.save(os.path.join(load_rate_path, save_word_list[1]), np.array(character_deleting_list))
    np.save(os.path.join(load_rate_path, save_word_list[2]), np.array(symbol_insertion_list))
    np.save(os.path.join(load_rate_path, save_word_list[3]), np.array(glyph_replacement_list))
    np.save(os.path.join(load_rate_path, save_word_list[4]), np.array(character_splitting_list))
    np.save(os.path.join(load_rate_path, save_word_list[5]), np.array(homophone_replacement_list))
    np.save(os.path.join(load_rate_path, save_word_list[6]), np.array(nasal_replacement_list))
    np.save(os.path.join(load_rate_path, save_word_list[7]), np.array(dorsal_replacement_list))
    np.save(os.path.join(load_rate_path, save_word_list[8]), np.array(context_prediction_list))
    np.save(os.path.join(load_rate_path, save_word_list[9]), np.array(synonym_replacement_list))
    np.save(os.path.join(load_rate_path, save_word_list[10]), np.array(traditional_conversion_list))

    word_shuffling_list = np.load(os.path.join(load_rate_path, save_word_list[0]), allow_pickle=True)
    character_deleting_list = np.load(os.path.join(load_rate_path, save_word_list[1]), allow_pickle=True)
    symbol_insertion_list = np.load(os.path.join(load_rate_path, save_word_list[2]), allow_pickle=True)
    glyph_replacement_list = np.load(os.path.join(load_rate_path, save_word_list[3]), allow_pickle=True)
    character_splitting_list = np.load(os.path.join(load_rate_path, save_word_list[4]), allow_pickle=True)
    homophone_replacement_list = np.load(os.path.join(load_rate_path, save_word_list[5]), allow_pickle=True)
    nasal_replacement_list = np.load(os.path.join(load_rate_path, save_word_list[6]), allow_pickle=True)
    dorsal_replacement_list = np.load(os.path.join(load_rate_path, save_word_list[7]), allow_pickle=True)
    context_prediction_list = np.load(os.path.join(load_rate_path, save_word_list[8]), allow_pickle=True)
    synonym_replacement_list = np.load(os.path.join(load_rate_path, save_word_list[9]), allow_pickle=True)
    traditional_conversion_list = np.load(os.path.join(load_rate_path, save_word_list[10]), allow_pickle=True)

    save_name_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                 'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                 'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                 'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                 'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                 'traditional_conversion_candidate_mutants.csv']
    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, word_shuffling_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[0])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, character_deleting_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[1])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, symbol_insertion_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[2])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, glyph_replacement_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[3])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, character_splitting_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[4])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, homophone_replacement_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[5])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, nasal_replacement_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[6])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, dorsal_replacement_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[7])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, context_prediction_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[8])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, synonym_replacement_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[9])

    sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list, span2_end_list, ori_strs, rep_strs \
        = get_sentences(test_data, select_words, select_words_positions, traditional_conversion_list)
    save_candidate_mutants(test_data, sentences, label_list, span1_begin_list, span1_end_list, span2_begin_list,
                           span2_end_list, ori_strs, rep_strs, load_rate_path, save_name_list[10])

if __name__ == '__main__':
    pass


