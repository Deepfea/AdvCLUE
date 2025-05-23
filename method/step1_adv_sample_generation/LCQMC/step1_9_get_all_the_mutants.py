import os
import pandas as pd


transferability_mutants_list = ['transferability_word_shuffling_mutants.csv',
                                    'transferability_character_deleting_mutants.csv',
                                    'transferability_symbol_insertion_mutants.csv',
                                    'transferability_glyph_replacement_mutants.csv',
                                    'transferability_character_splitting_mutants.csv',
                                    'transferability_homophone_replacement_mutants.csv',
                                    'transferability_nasal_replacement_mutants.csv',
                                    'transferability_dorsal_replacement_mutants.csv',
                                    'transferability_context_prediction_mutants.csv',
                                    'transferability_synonym_replacement_mutants.csv',
                                    'transferability_traditional_conversion_mutants.csv']

def save_all_the_mutants(load_path):
    flag = 0
    for name_num in range(len(transferability_mutants_list)):
        print(os.path.join(load_path, transferability_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, transferability_mutants_list[name_num]))
        if flag == 0:
            final_mutants = candidate_mutants
            flag = 1
        else:
            final_mutants = pd.concat([final_mutants, candidate_mutants], ignore_index=True)
    if flag == 0:
        print("一个变异成功的数据也没有！！！")
    else:
        print(len(final_mutants))
        final_mutants.to_csv(os.path.join(load_path, 'final_mutants.csv'), index=False)


if __name__ == '__main__':
    pass
