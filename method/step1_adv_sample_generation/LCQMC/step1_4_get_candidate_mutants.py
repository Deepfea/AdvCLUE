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
from method.step1_adv_sample_generation.LCQMC.get_sentence import get_sentence

def save_candidate_mutants(test_data, mutant_x, mutant_y, ori_strs, mut_strs, load_rate_path, save_name):
    ori_text_a_list = []
    ori_text_b_list = []
    text_a_list = []
    text_b_list = []
    str_list = []
    word_list = []
    label_list = []
    for data_num in range(len(test_data)):
        label = test_data.loc[data_num, 'label']
        ori_text_a = test_data.loc[data_num, 'text_a']
        ori_text_b = test_data.loc[data_num, 'text_b']
        text_a = mutant_x[data_num]
        text_b = mutant_y[data_num]
        ori_str = ori_strs[data_num]
        mut_str =mut_strs[data_num]

        ori_text_a_list.append(ori_text_a)
        ori_text_b_list.append(ori_text_b)
        text_a_list.append(text_a)
        text_b_list.append(text_b)
        str_list.append(ori_str)
        word_list.append(mut_str)
        label_list.append(label)
    merge_dt_dict = {'ori_text_a': ori_text_a_list, 'ori_text_b': ori_text_b_list, 'text_a': text_a_list, 'text_b': text_b_list,
                     'str': str_list, 'word': word_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    save_path = os.path.join(load_rate_path, save_name)
    data_df.to_csv(save_path, index=False)


def get_sentences(test_data, select_words, select_word_texts, mutate_list):
    text_a_list = []
    text_b_list = []
    ori_strs = []
    rep_strs = []
    for data_num in range(len(select_words)):
        text_a = test_data.loc[data_num, 'text_a']
        text_b = test_data.loc[data_num, 'text_b']
        words = select_words[data_num]
        word_text = select_word_texts[data_num]
        x_a, x_b, y, z = get_sentence(text_a, text_b, words, word_text, mutate_list[data_num])
        text_a_list.append(x_a)
        text_b_list.append(x_b)
        ori_strs.append(y)
        rep_strs.append(z)
    return text_a_list, text_b_list, ori_strs, rep_strs

def generate_mutant(data_path, rate):
    print(rate)
    load_rate_path = os.path.join(data_path, str(rate))
    select_words = np.load(os.path.join(load_rate_path, 'select_words.npy'), allow_pickle=True)
    select_word_texts = np.load(os.path.join(load_rate_path, 'select_word_texts.npy'), allow_pickle=True)
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
    # fact_list = []
    for data_num in tqdm(range(len(test_data))):
        seg_list = list(select_words[data_num])
        fact_a = test_data.loc[data_num, 'text_a']
        fact_b = test_data.loc[data_num, 'text_b']
        # fact_list.append(fact)
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
            if select_word_texts[data_num][seg_num] == 'a':
                fact = fact_a
            else:
                fact = fact_b
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


    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, word_shuffling_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[0])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, character_deleting_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[1])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, symbol_insertion_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[2])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, glyph_replacement_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[3])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, character_splitting_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[4])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, homophone_replacement_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[5])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, nasal_replacement_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[6])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, dorsal_replacement_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[7])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, context_prediction_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[8])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, synonym_replacement_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[9])

    sentence_x, sentence_y, ori_strs, rep_strs = get_sentences(test_data, select_words, select_word_texts, traditional_conversion_list)
    save_candidate_mutants(test_data, sentence_x, sentence_y, ori_strs, rep_strs,
                           load_rate_path, save_name_list[10])

if __name__ == '__main__':
    pass
