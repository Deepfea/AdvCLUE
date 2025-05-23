import numpy as np
from bert_score import score
import os
import pandas as pd


final_mutants_list = ['word_shuffling_mutants.csv', 'character_deleting_mutants.csv',
                      'symbol_insertion_mutants.csv', 'glyph_replacement_mutants.csv',
                      'character_splitting_mutants.csv', 'homophone_replacement_mutants.csv',
                      'nasal_replacement_mutants.csv', 'dorsal_replacement_mutants.csv',
                      'context_prediction_mutants.csv', 'synonym_replacement_mutants.csv',
                      'traditional_conversion_mutants.csv']
fidelity_mutants_list = ['fidelity_word_shuffling_mutants.csv', 'fidelity_character_deleting_mutants.csv',
                      'fidelity_symbol_insertion_mutants.csv', 'fidelity_glyph_replacement_mutants.csv',
                      'fidelity_character_splitting_mutants.csv', 'fidelity_homophone_replacement_mutants.csv',
                      'fidelity_nasal_replacement_mutants.csv', 'fidelity_dorsal_replacement_mutants.csv',
                      'fidelity_context_prediction_mutants.csv', 'fidelity_synonym_replacement_mutants.csv',
                      'fidelity_traditional_conversion_mutants.csv']
mut_name_list = ['word_shuffling', 'character_deleting',
                       'symbol_insertion', 'glyph_replacement',
                       'character_splitting', 'homophone_replacement',
                       'nasal_replacement', 'dorsal_replacement',
                       'context_prediction', 'synonym_replacement',
                       'traditional_conversion']
fidelity_mut_name_list = ['fidelity_word_shuffling', 'fidelity_character_deleting',
                       'fidelity_symbol_insertion', 'fidelity_glyph_replacement',
                       'fidelity_character_splitting', 'fidelity_homophone_replacement',
                       'fidelity_nasal_replacement', 'fidelity_dorsal_replacement',
                       'fidelity_context_prediction', 'fidelity_synonym_replacement',
                       'fidelity_traditional_conversion']
def get_fidelity_mutants(load_path):
    stan_score = 0.85
    for name_num in range(len(final_mutants_list)):
        print(os.path.join(load_path, final_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, final_mutants_list[name_num]))
        ori_text = list(candidate_mutants['ori_text'])
        text = list(candidate_mutants['text'])
        p, r, f1_list = score(text, ori_text, model_type='bert-base-chinese', lang='zh', verbose=False)
        f1_list = f1_list.cpu().numpy()
        test_candidates = np.load(os.path.join(load_path, mut_name_list[name_num] + '_mutant_candidate.npy'),
                                  allow_pickle=True)
        print(len(test_candidates))
        print(len(f1_list))
        test_candidate_ids = np.load(os.path.join(load_path, mut_name_list[name_num] + '_mutant_candidate_ids.npy'),
                                     allow_pickle=True)
        belong_list = []
        tag_list = []
        ori_text_list = []
        str_list = []
        word_list = []
        text_list = []
        candidate_list = []
        groundTruth_list = []
        label_list = []
        mut_list = []
        mutant_candidates = []
        mutant_candidate_ids = []

        for data_num in range(len(f1_list)):
            if f1_list[data_num] < stan_score:
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            tag_list.append(candidate_mutants.loc[data_num, 'tag'])
            ori_text_list.append(candidate_mutants.loc[data_num, 'ori_text'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            candidate_list.append(candidate_mutants.loc[data_num, 'candidate'])
            groundTruth_list.append(candidate_mutants.loc[data_num, 'groundTruth'])
            label_list.append(candidate_mutants.loc[data_num, 'label'])
            mut_list.append(candidate_mutants.loc[data_num, 'mut'])
            mutant_candidates.append(test_candidates[data_num])
            mutant_candidate_ids.append(test_candidate_ids[data_num])

        merge_dt_dict = {'belong': belong_list, 'tag': tag_list, 'ori_text': ori_text_list, 'str': str_list, 'word': word_list,
                         'text': text_list, 'candidate': candidate_list, 'groundTruth': groundTruth_list,
                         'label': label_list, 'type': mut_list}

        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, fidelity_mutants_list[name_num]), index=False)

        mutant_candidates = np.array(mutant_candidates)
        mutant_candidate_ids = np.array(mutant_candidate_ids)

        np.save(os.path.join(load_path, fidelity_mut_name_list[name_num] + '_mutant_candidate.npy'), mutant_candidates)
        np.save(os.path.join(load_path, fidelity_mut_name_list[name_num] + '_mutant_candidate_ids.npy'), mutant_candidate_ids)

        print(len(data_df))


if __name__ == '__main__':
    pass
