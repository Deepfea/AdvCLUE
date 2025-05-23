
import os

import numpy as np
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

transferability_mut_name_list = ['transferability_word_shuffling', 'transferability_character_deleting',
                       'transferability_symbol_insertion', 'transferability_glyph_replacement',
                       'transferability_character_splitting', 'transferability_homophone_replacement',
                       'transferability_nasal_replacement', 'transferability_dorsal_replacement',
                       'transferability_context_prediction', 'transferability_synonym_replacement',
                       'transferability_traditional_conversion']


def save_all_the_mutants(load_path):
    flag = 0
    for name_num in range(len(transferability_mutants_list)):
        print(os.path.join(load_path, transferability_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, transferability_mutants_list[name_num]))
        candidate_mutant_candidates = np.load(os.path.join(load_path, transferability_mut_name_list[name_num] + '_mutant_candidate.npy'))
        candidate_mutant_candidate_ids = np.load(os.path.join(load_path, transferability_mut_name_list[name_num] + '_mutant_candidate_ids.npy'))

        if flag == 0:
            final_mutants = candidate_mutants
            final_mutant_candidates = candidate_mutant_candidates
            final_mutant_candidate_ids = candidate_mutant_candidate_ids
            flag = 1
        else:
            final_mutants = pd.concat([final_mutants, candidate_mutants], ignore_index=True)
            final_mutant_candidates = np.concatenate((final_mutant_candidates, candidate_mutant_candidates), axis=0)
            final_mutant_candidate_ids = np.concatenate((final_mutant_candidate_ids, candidate_mutant_candidate_ids), axis=0)
        print(len(final_mutants))
        print(len(final_mutant_candidates))
        print(len(final_mutant_candidate_ids))
    if flag == 0:
        print("一个变异成功的数据也没有！！！")
    else:
        print(len(final_mutants))
        final_mutants.to_csv(os.path.join(load_path, 'final_mutants.csv'), index=False)
        np.save(os.path.join(load_path, 'final_mutant_candidate.npy'), final_mutant_candidates)
        np.save(os.path.join(load_path, 'final_mutant_candidate_ids.npy'), final_mutant_candidate_ids)

if __name__ == '__main__':
    pass
