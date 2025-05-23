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
from method.step1_adv_sample_generation.PKU.get_sentence import get_sentence

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

def save_candidate_mutants(test_data, sentences_x, sentences_y, ori_strs, mut_strs, load_rate_path, save_name):
    max_len = 500
    sep_word = '@'  # 拆分句子的文本分隔符
    sep_label = 'S'  # 拆分句子的标签分隔符
    del_string = '\uf7ee'
    x = []
    y = []
    for num in range(len(sentences_x)):
        if del_string in sentences_x[num]:
            sentences_x[num], sentences_y[num] = del_str(sentences_x[num], sentences_y[num])
        if len(sentences_x[num]) > max_len:
            sub_word_list = get_sub_list(sentences_x[num], max_len - 5, sep_word)
            sub_label_list = get_sub_list(sentences_y[num], max_len - 5, sep_label)
            x.append(sub_word_list[0])
            y.append(sub_label_list[0])
        else:
            x.append(sentences_x[num])
            y.append(sentences_y[num])
    output_dir = os.path.join(load_rate_path, save_name)
    print(output_dir)
    np.savez_compressed(output_dir, data=test_data['words'], data_label=test_data['labels'],
                        mutants=x, mutants_label=y, strs=ori_strs, words=mut_strs)

def get_sentences(test_data, select_words, select_words_positions, mutate_list):
    x_test = test_data['words']
    y_test = test_data['labels']
    ori_strs = []
    rep_strs = []
    sentences_x = []
    sentences_y = []
    for data_num in range(len(select_words)):
        temp_x = x_test[data_num]
        temp_y = y_test[data_num]
        words = select_words[data_num]
        words_position = select_words_positions[data_num]
        temp_mutate_list = mutate_list[data_num]
        x, y, z1,  z2 = get_sentence(temp_x, temp_y, words, words_position, temp_mutate_list)
        sentences_x.append(x)
        sentences_y.append(y)
        ori_strs.append(z1)
        rep_strs.append(z2)
    return sentences_x, sentences_y, ori_strs, rep_strs

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
    select_words_positions = np.load(os.path.join(load_rate_path, 'select_words_positions.npy'), allow_pickle=True)

    test_dataset_path = os.path.join(data_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    fact_list = get_fact(x_test)

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

    for data_num in tqdm(range(len(fact_list))):
        seg_list = list(select_words[data_num])
        fact = fact_list[data_num]
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

    save_name_list = ['word_shuffling_candidate_mutants', 'character_deleting_candidate_mutants',
                 'symbol_insertion_candidate_mutants', 'glyph_replacement_candidate_mutants',
                 'character_splitting_candidate_mutants', 'homophone_replacement_candidate_mutants',
                 'nasal_replacement_candidate_mutants', 'dorsal_replacement_candidate_mutants',
                 'context_prediction_candidate_mutants', 'synonym_replacement_candidate_mutants',
                 'traditional_conversion_candidate_mutants']
    word_shuffling_sentences_x, word_shuffling_sentences_y, word_shuffling_ori_strs, word_shuffling_rep_strs = get_sentences(test_data, select_words, select_words_positions, word_shuffling_list)
    save_candidate_mutants(test_data, word_shuffling_sentences_x, word_shuffling_sentences_y, word_shuffling_ori_strs, word_shuffling_rep_strs,
                           load_rate_path, save_name_list[0])

    character_deleting_sentences_x, character_deleting_sentences_y, character_deleting_ori_strs, character_deleting_rep_strs = get_sentences(test_data, select_words, select_words_positions, character_deleting_list)
    save_candidate_mutants(test_data, character_deleting_sentences_x, character_deleting_sentences_y, character_deleting_ori_strs, character_deleting_rep_strs,
                           load_rate_path, save_name_list[1])

    symbol_insertion_sentences_x, symbol_insertion_sentences_y, symbol_insertion_ori_strs, symbol_insertion_rep_strs = get_sentences(test_data, select_words, select_words_positions, symbol_insertion_list)
    save_candidate_mutants(test_data, symbol_insertion_sentences_x, symbol_insertion_sentences_y, symbol_insertion_ori_strs, symbol_insertion_rep_strs,
                           load_rate_path, save_name_list[2])

    glyph_replacement_sentences_x, glyph_replacement_sentences_y, glyph_replacement_ori_strs, glyph_replacement_rep_strs = get_sentences(test_data, select_words, select_words_positions, glyph_replacement_list)
    save_candidate_mutants(test_data, glyph_replacement_sentences_x, glyph_replacement_sentences_y, glyph_replacement_ori_strs, glyph_replacement_rep_strs,
                           load_rate_path, save_name_list[3])

    character_splitting_sentences_x, character_splitting_sentences_y, character_splitting_ori_strs, character_splitting_rep_strs = get_sentences(test_data, select_words, select_words_positions, character_splitting_list)
    save_candidate_mutants(test_data, character_splitting_sentences_x, character_splitting_sentences_y, character_splitting_ori_strs, character_splitting_rep_strs,
                           load_rate_path, save_name_list[4])

    homophone_replacement_sentences_x, homophone_replacement_sentences_y, homophone_replacement_ori_strs, homophone_replacement_rep_strs = get_sentences(test_data, select_words, select_words_positions, homophone_replacement_list)
    save_candidate_mutants(test_data, homophone_replacement_sentences_x, homophone_replacement_sentences_y, homophone_replacement_ori_strs, homophone_replacement_rep_strs,
                           load_rate_path, save_name_list[5])

    nasal_replacement_sentences_x, nasal_replacement_sentences_y, nasal_replacement_ori_strs, nasal_replacement_rep_strs = get_sentences(test_data, select_words, select_words_positions, nasal_replacement_list)
    save_candidate_mutants(test_data, nasal_replacement_sentences_x, nasal_replacement_sentences_y, nasal_replacement_ori_strs, nasal_replacement_rep_strs,
                           load_rate_path, save_name_list[6])

    dorsal_replacement_sentences_x, dorsal_replacement_sentences_y, dorsal_replacement_ori_strs, dorsal_replacement_rep_strs = get_sentences(test_data, select_words, select_words_positions, dorsal_replacement_list)
    save_candidate_mutants(test_data, dorsal_replacement_sentences_x, dorsal_replacement_sentences_y, dorsal_replacement_ori_strs, dorsal_replacement_rep_strs,
                           load_rate_path, save_name_list[7])

    context_prediction_sentences_x, context_prediction_sentences_y, context_prediction_ori_strs, context_prediction_rep_strs = get_sentences(test_data, select_words, select_words_positions, context_prediction_list)
    save_candidate_mutants(test_data, context_prediction_sentences_x, context_prediction_sentences_y, context_prediction_ori_strs, context_prediction_rep_strs,
                           load_rate_path, save_name_list[8])

    synonym_replacement_sentences_x, synonym_replacement_sentences_y, synonym_replacement_ori_strs, synonym_replacement_rep_strs = get_sentences(test_data, select_words, select_words_positions, synonym_replacement_list)
    save_candidate_mutants(test_data, synonym_replacement_sentences_x, synonym_replacement_sentences_y, synonym_replacement_ori_strs, synonym_replacement_rep_strs,
                           load_rate_path, save_name_list[9])

    traditional_conversion_sentences_x, traditional_conversion_sentences_y, traditional_conversion_ori_strs, traditional_conversion_rep_strs = get_sentences(test_data, select_words, select_words_positions, traditional_conversion_list)
    save_candidate_mutants(test_data, traditional_conversion_sentences_x, traditional_conversion_sentences_y, traditional_conversion_ori_strs, traditional_conversion_rep_strs,
                           load_rate_path, save_name_list[10])

if __name__ == '__main__':
    pass
