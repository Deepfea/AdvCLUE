from bert_score import score
import numpy as np
import os


final_mutants_list = ['word_shuffling_mutants.npz', 'character_deleting_mutants.npz',
                        'symbol_insertion_mutants.npz', 'glyph_replacement_mutants.npz',
                        'character_splitting_mutants.npz', 'homophone_replacement_mutants.npz',
                        'nasal_replacement_mutants.npz', 'dorsal_replacement_mutants.npz',
                        'context_prediction_mutants.npz', 'synonym_replacement_mutants.npz',
                        'traditional_conversion_mutants.npz']
fidelity_mutants_list = ['fidelity_word_shuffling_mutants.npz', 'fidelity_character_deleting_mutants.npz',
                      'fidelity_symbol_insertion_mutants.npz', 'fidelity_glyph_replacement_mutants.npz',
                      'fidelity_character_splitting_mutants.npz', 'fidelity_homophone_replacement_mutants.npz',
                      'fidelity_nasal_replacement_mutants.npz', 'fidelity_dorsal_replacement_mutants.npz',
                      'fidelity_context_prediction_mutants.npz', 'fidelity_synonym_replacement_mutants.npz',
                      'fidelity_traditional_conversion_mutants.npz']

def get_data_sentences(candidate_mutants):
    ori_texts = []
    texts = []
    for num in range(len(candidate_mutants['data'])):
        word_list = candidate_mutants['data'][num]
        temp_str = ''
        for word_num in range(len(word_list)):
            temp_str += word_list[word_num]
        ori_texts.append(temp_str)

        word_list = candidate_mutants['mutants'][num]
        temp_str = ''
        for word_num in range(len(word_list)):
            temp_str += word_list[word_num]
        texts.append(temp_str)
    return ori_texts, texts

def get_fidelity_mutants(load_path):
    stan_score = 0.85
    for name_num in range(len(final_mutants_list)):
        print(os.path.join(load_path, final_mutants_list[name_num]))
        candidate_mutants = np.load(os.path.join(load_path, final_mutants_list[name_num]), allow_pickle=True)
        ori_text, text = get_data_sentences(candidate_mutants)
        p, r, f1_list = score(text, ori_text, model_type='bert-base-chinese', lang='zh', verbose=False)
        f1_list = f1_list.cpu().numpy()

        belong_list = []
        ori_text_list = []
        ori_label_list = []
        text_list = []
        label_list = []
        str_list = []
        word_list = []
        pre_list = []
        mut_list = []
        for data_num in range(len(f1_list)):
            if f1_list[data_num] < stan_score:
                continue
            belong_list.append(candidate_mutants['belong'][data_num])
            ori_text_list.append(candidate_mutants['data'][data_num])
            ori_label_list.append(candidate_mutants['data_label'][data_num])
            text_list.append(candidate_mutants['mutants'][data_num])
            label_list.append(candidate_mutants['mutants_label'][data_num])
            str_list.append(candidate_mutants['strs'][data_num])
            word_list.append(candidate_mutants['words'][data_num])
            pre_list.append(candidate_mutants['pre'][data_num])
            mut_list.append(candidate_mutants['mut'][data_num])

        np.savez_compressed(os.path.join(load_path, fidelity_mutants_list[name_num]), belong=belong_list, ori_text=ori_text_list,
                            ori_label=ori_label_list, text=text_list, label=label_list,
                            str=str_list, word=word_list, prediction=pre_list, type=mut_list)
        print(len(belong_list))


if __name__ == '__main__':
    pass
