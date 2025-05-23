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
        ori_text_a = list(candidate_mutants['ori_context'])
        ori_text_b = list(candidate_mutants['ori_question'])
        text_a = list(candidate_mutants['context'])
        text_b = list(candidate_mutants['question'])
        p, r, f1_list1 = score(text_a, ori_text_a, model_type='bert-base-chinese', lang='zh', verbose=False)
        p, r, f1_list2 = score(text_b, ori_text_b, model_type='bert-base-chinese', lang='zh', verbose=False)
        f1_list1 = f1_list1.cpu().numpy()
        f1_list2 = f1_list2.cpu().numpy()

        belong_list = []
        ori_context_list = []
        ori_question_list = []
        ori_answer_start_list = []
        ori_answer_end_list = []
        answer_list = []
        context_list = []
        question_list = []
        answer_start_list = []
        answer_end_list = []
        str_list = []
        word_list = []
        text_list = []
        mut_list = []

        for data_num in range(len(f1_list1)):
            if f1_list1[data_num] < stan_score or f1_list2[data_num] < stan_score:
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            ori_context_list.append(candidate_mutants.loc[data_num, 'ori_context'])
            ori_question_list.append(candidate_mutants.loc[data_num, 'ori_question'])
            ori_answer_start_list.append(candidate_mutants.loc[data_num, 'ori_answer_start'])
            ori_answer_end_list.append(candidate_mutants.loc[data_num, 'ori_answer_end'])
            answer_list.append(candidate_mutants.loc[data_num, 'answer'])
            context_list.append(candidate_mutants.loc[data_num, 'context'])
            question_list.append(candidate_mutants.loc[data_num, 'question'])
            answer_start_list.append(candidate_mutants.loc[data_num, 'answer_start'])
            answer_end_list.append(candidate_mutants.loc[data_num, 'answer_end'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            text_list.append(candidate_mutants.loc[data_num, 'text'])
            mut_list.append(candidate_mutants.loc[data_num, 'mut'])

        merge_dt_dict = {'belong': belong_list, 'ori_context': ori_context_list, 'ori_question': ori_question_list,
                         'ori_answer_start': ori_answer_start_list, 'ori_answer_end': ori_answer_end_list, 'answer': answer_list,
                         'context': context_list, 'question': question_list, 'answer_start': answer_start_list,
                         'answer_end': answer_end_list, 'str': str_list, 'word': word_list, 'text': text_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, fidelity_mutants_list[name_num]), index=False)
        print(len(data_df))


if __name__ == '__main__':
    pass
