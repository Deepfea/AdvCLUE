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

def get_fidelity_mutants(load_path):
    stan_score = 0.85
    for name_num in range(len(final_mutants_list)):
        print(os.path.join(load_path, final_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, final_mutants_list[name_num]))
        ori_text = list(candidate_mutants['ori_text'])
        text = list(candidate_mutants['text'])
        p, r, f1_list = score(text, ori_text, model_type='bert-base-chinese', lang='zh', verbose=False)
        f1_list = f1_list.cpu().numpy()

        belong_list = []
        ori_text_list = []
        str_list = []
        word_list = []
        text_list = []
        question_list = []
        candidate_0_list = []
        candidate_1_list = []
        candidate_2_list = []
        candidate_3_list = []
        answer_list = []
        mut_list = []

        for data_num in range(len(f1_list)):
            if f1_list[data_num] < stan_score:
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            ori_text_list.append(candidate_mutants.loc[data_num, 'ori_text'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            question_list.append(candidate_mutants.loc[data_num, 'question'])
            candidate_0_list.append(candidate_mutants.loc[data_num, 'candidate_0'])
            candidate_1_list.append(candidate_mutants.loc[data_num, 'candidate_1'])
            candidate_2_list.append(candidate_mutants.loc[data_num, 'candidate_2'])
            candidate_3_list.append(candidate_mutants.loc[data_num, 'candidate_3'])
            answer_list.append(candidate_mutants.loc[data_num, 'answer'])
            mut_list.append(candidate_mutants.loc[data_num, 'mut'])
        merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list,
                         'word': word_list, 'text': text_list, 'question': question_list,
                         'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list, 'candidate_2': candidate_2_list,
                         'candidate_3': candidate_3_list, 'answer': answer_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, fidelity_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
