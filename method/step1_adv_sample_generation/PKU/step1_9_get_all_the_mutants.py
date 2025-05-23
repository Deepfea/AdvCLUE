import os
import numpy as np

transferability_mutants_list = ['transferability_word_shuffling_mutants.npz', 'transferability_character_deleting_mutants.npz',
                      'transferability_symbol_insertion_mutants.npz', 'transferability_glyph_replacement_mutants.npz',
                      'transferability_character_splitting_mutants.npz', 'transferability_homophone_replacement_mutants.npz',
                      'transferability_nasal_replacement_mutants.npz', 'transferability_dorsal_replacement_mutants.npz',
                      'transferability_context_prediction_mutants.npz', 'transferability_synonym_replacement_mutants.npz',
                      'transferability_traditional_conversion_mutants.npz']

def save_all_the_mutants(load_path):
    belong_list = []
    ori_text_list = []
    ori_label_list = []
    text_list = []
    label_list = []
    str_list = []
    word_list = []
    mut_list = []
    for mut_num in range(len(transferability_mutants_list)):
        mutant_path = os.path.join(load_path, transferability_mutants_list[mut_num])
        temp_mutants = np.load(mutant_path, allow_pickle=True)
        belong_list.extend(temp_mutants['belong'])
        ori_text_list.extend(temp_mutants['ori_text'])
        ori_label_list.extend(temp_mutants['ori_label'])
        text_list.extend(temp_mutants['text'])
        label_list.extend(temp_mutants['label'])
        str_list.extend(temp_mutants['str'])
        word_list.extend(temp_mutants['word'])
        mut_list.extend(temp_mutants['type'])

    if len(belong_list) == 0:
        print("一个变异成功的数据也没有！！！")
    else:
        print(len(belong_list))
        np.savez_compressed(os.path.join(load_path, 'final_mutants.npz'), belong=belong_list,
                            ori_text=ori_text_list, ori_label=ori_label_list, text=text_list, label=label_list,
                            str=str_list, word=word_list, type=mut_list)

if __name__ == '__main__':
    pass
