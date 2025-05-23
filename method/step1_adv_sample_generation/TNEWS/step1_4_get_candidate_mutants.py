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
from method.step1_adv_sample_generation.TNEWS.get_sentence import get_sentence

def save_candidate_mutants(test_data, mutants, ori_strs, mut_strs, load_rate_path, save_name):
    fact_list = []
    mutant_list = []
    str_list = []
    word_list = []
    label_list = []
    for data_num in range(len(test_data)):
        label = test_data.loc[data_num, 'label']
        fact = test_data.loc[data_num, 'text']
        mutant = mutants[data_num]
        ori_str = ori_strs[data_num]
        mut_str =mut_strs[data_num]
        fact_list.append(fact)
        mutant_list.append(mutant)
        str_list.append(ori_str)
        word_list.append(mut_str)
        label_list.append(label)
    merge_dt_dict = {'text': fact_list, 'str': str_list, 'word': word_list, 'mutant': mutant_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    save_path = os.path.join(load_rate_path, save_name)
    data_df.to_csv(save_path, index=False)


def get_sentences(fact_list, select_words, mutate_list):
    sentences = []
    ori_strs = []
    rep_strs = []
    for data_num in range(len(select_words)):
        fact = fact_list[data_num]
        words = select_words[data_num]
        x, y, z = get_sentence(fact, words, mutate_list[data_num])
        sentences.append(x)
        ori_strs.append(y)
        rep_strs.append(z)
    return sentences, ori_strs, rep_strs

def generate_mutant(data_path, rate):
    print(rate)
    load_rate_path = os.path.join(data_path, str(rate))
    select_words = np.load(os.path.join(load_rate_path, 'select_words.npy'), allow_pickle=True)
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
    fact_list = []
    for data_num in tqdm(range(len(test_data))):
        seg_list = list(select_words[data_num])
        fact = test_data.loc[data_num, 'text']
        fact_list.append(fact)
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
    save_name_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                 'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                 'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                 'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                 'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                 'traditional_conversion_candidate_mutants.csv']
    word_shuffling_sentences, word_shuffling_ori_strs, word_shuffling_rep_strs = get_sentences(fact_list, select_words,word_shuffling_list)
    save_candidate_mutants(test_data, word_shuffling_sentences, word_shuffling_ori_strs, word_shuffling_rep_strs,
                           load_rate_path, save_name_list[0])

    character_deleting_sentences, character_deleting_ori_strs, character_deleting_rep_strs = get_sentences(fact_list, select_words, character_deleting_list)
    save_candidate_mutants(test_data, character_deleting_sentences, character_deleting_ori_strs, character_deleting_rep_strs,
                           load_rate_path, save_name_list[1])

    symbol_insertion_sentences, symbol_insertion_ori_strs, symbol_insertion_rep_strs = get_sentences(fact_list, select_words, symbol_insertion_list)
    save_candidate_mutants(test_data, symbol_insertion_sentences, symbol_insertion_ori_strs, symbol_insertion_rep_strs,
                           load_rate_path, save_name_list[2])

    glyph_replacement_sentences, glyph_replacement_ori_strs, glyph_replacement_rep_strs = get_sentences(fact_list, select_words, glyph_replacement_list)
    save_candidate_mutants(test_data, glyph_replacement_sentences, glyph_replacement_ori_strs, glyph_replacement_rep_strs,
                           load_rate_path, save_name_list[3])

    character_splitting_sentences, character_splitting_ori_strs, character_splitting_rep_strs = get_sentences(fact_list, select_words, character_splitting_list)
    save_candidate_mutants(test_data, character_splitting_sentences, character_splitting_ori_strs, character_splitting_rep_strs,
                           load_rate_path, save_name_list[4])

    homophone_replacement_sentences, homophone_replacement_ori_strs, homophone_replacement_rep_strs = get_sentences(fact_list, select_words, homophone_replacement_list)
    save_candidate_mutants(test_data, homophone_replacement_sentences, homophone_replacement_ori_strs, homophone_replacement_rep_strs,
                           load_rate_path, save_name_list[5])

    nasal_replacement_sentences, nasal_replacement_ori_strs, nasal_replacement_rep_strs = get_sentences(fact_list, select_words, nasal_replacement_list)
    save_candidate_mutants(test_data, nasal_replacement_sentences, nasal_replacement_ori_strs, nasal_replacement_rep_strs,
                           load_rate_path, save_name_list[6])

    dorsal_replacement_sentences, dorsal_replacement_ori_strs, dorsal_replacement_rep_strs = get_sentences(fact_list, select_words, dorsal_replacement_list)
    save_candidate_mutants(test_data, dorsal_replacement_sentences, dorsal_replacement_ori_strs, dorsal_replacement_rep_strs,
                           load_rate_path, save_name_list[7])

    context_prediction_sentences, context_prediction_ori_strs, context_prediction_rep_strs = get_sentences(fact_list, select_words, context_prediction_list)
    save_candidate_mutants(test_data, context_prediction_sentences, context_prediction_ori_strs, context_prediction_rep_strs,
                           load_rate_path, save_name_list[8])

    synonym_replacement_sentences, synonym_replacement_ori_strs, synonym_replacement_rep_strs = get_sentences(fact_list, select_words, synonym_replacement_list)
    save_candidate_mutants(test_data, synonym_replacement_sentences, synonym_replacement_ori_strs, synonym_replacement_rep_strs,
                           load_rate_path, save_name_list[9])

    traditional_conversion_sentences, traditional_conversion_ori_strs, traditional_conversion_rep_strs = get_sentences(fact_list, select_words, traditional_conversion_list)
    save_candidate_mutants(test_data, traditional_conversion_sentences, traditional_conversion_ori_strs, traditional_conversion_rep_strs,
                           load_rate_path, save_name_list[10])

if __name__ == '__main__':
    pass
