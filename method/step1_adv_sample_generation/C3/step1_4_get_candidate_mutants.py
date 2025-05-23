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
from method.step1_adv_sample_generation.C3.get_sentence import get_sentence

def save_candidate_mutants(test_data, mutant, ori_strs, mut_strs, load_rate_path, save_name):
    ori_text_list = []
    text_list = []
    question_list = []
    candidate_0_list = []
    candidate_1_list = []
    candidate_2_list = []
    candidate_3_list = []
    answer_list = []
    str_list = []
    word_list = []

    for data_num in range(len(test_data)):
        ori_text = test_data.loc[data_num, 'text']
        question = test_data.loc[data_num, 'question']
        text = mutant[data_num]
        ori_str = ori_strs[data_num]
        mut_str = mut_strs[data_num]
        candidate_0 = test_data.loc[data_num, 'candidate_0']
        candidate_1 = test_data.loc[data_num, 'candidate_1']
        candidate_2 = test_data.loc[data_num, 'candidate_2']
        candidate_3 = test_data.loc[data_num, 'candidate_3']
        answer = test_data.loc[data_num, 'answer']

        ori_text_list.append(ori_text)
        question_list.append(question)
        text_list.append(text)
        str_list.append(ori_str)
        word_list.append(mut_str)
        candidate_0_list.append(candidate_0)
        candidate_1_list.append(candidate_1)
        candidate_2_list.append(candidate_2)
        candidate_3_list.append(candidate_3)
        answer_list.append(answer)

    merge_dt_dict = {'ori_text': ori_text_list, 'question': question_list, 'text': text_list,
                     'str': str_list, 'word': word_list, 'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list,
                     'candidate_2': candidate_2_list, 'candidate_3': candidate_3_list, 'answer': answer_list}
    data_df = pd.DataFrame(merge_dt_dict)
    save_path = os.path.join(load_rate_path, save_name)
    data_df.to_csv(save_path, index=False)


def get_sentences(test_data, select_words, mutate_list):
    text_list = []
    ori_strs = []
    rep_strs = []
    for data_num in range(len(select_words)):
        text = test_data.loc[data_num, 'text']
        words = select_words[data_num]
        x, y, z = get_sentence(text, words, mutate_list[data_num])
        text_list.append(x)
        ori_strs.append(y)
        rep_strs.append(z)
    return text_list, ori_strs, rep_strs

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


    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, word_shuffling_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[0])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, character_deleting_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[1])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, symbol_insertion_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[2])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, glyph_replacement_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[3])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, character_splitting_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[4])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, homophone_replacement_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[5])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, nasal_replacement_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[6])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, dorsal_replacement_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[7])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, context_prediction_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[8])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, synonym_replacement_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[9])

    sentence, ori_strs, rep_strs = get_sentences(test_data, select_words, traditional_conversion_list)
    save_candidate_mutants(test_data, sentence, ori_strs, rep_strs,
                           load_rate_path, save_name_list[10])

if __name__ == '__main__':
    pass
