import numpy as np
import os
import torch
from tqdm import tqdm
import pandas as pd
from create_dataset import create_examples, get_dataset
import transformers
from transformers import BertTokenizer
from model import BertForClassification
from torch.utils.data import DataLoader, SequentialSampler

class Config:
    max_length = 512
    epochs = 8
    batch_size = 4
    n_class = 4
    load_path = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese'
    device = os.environ.get("DEVICE", "cuda:0")

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(config, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=eval_sampler)

    eval_loss, eval_accuracy = 0.0, 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = []

    for _, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=False)):
        model.eval()
        batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3] if len(batch) == 4 else None}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()
        label_ids = batch[3].cpu().numpy()

        for i in range(len(logits)):
            logits_all += [logits[i]]
        tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples
    print(eval_accuracy)
    return logits_all


def cal_output_label(test_df, base_model_path):
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.load_path)
    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(base_model_path, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    test_examples = create_examples(test_df)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    output_list = evaluate(config, model, test_dataset)
    pre_labels = []
    for num in range(len(output_list)):
        temp_arr = output_list[num]
        pre_labels.append(np.argmax(temp_arr))
    output_arr = np.array(pre_labels)

    return output_arr

def cal_candidate_output(base_model_path, load_path, rate):
    load_data_path = os.path.join(load_path, 'source_data', str(rate))
    load_name_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                      'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                      'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                      'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                      'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                      'traditional_conversion_candidate_mutants.csv']
    label_file_list = ['word_shuffling_labels.npy', 'character_deleting_labels.npy',
                       'symbol_insertion_labels.npy', 'glyph_replacement_labels.npy',
                       'character_splitting_labels.npy', 'homophone_replacement_labels.npy',
                       'nasal_replacement_labels.npy', 'dorsal_replacement_labels.npy',
                       'context_prediction_labels.npy', 'synonym_replacement_labels.npy',
                       'traditional_conversion_labels.npy']
    for name_num in range(len(load_name_list)):
        print(os.path.join(load_data_path, load_name_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_data_path, load_name_list[name_num]))
        labels = cal_output_label(candidate_mutants, base_model_path)
        np.save(os.path.join(load_data_path, label_file_list[name_num]), labels)

def select_final_mutants(save_path, model_path, source_data_path, rates):
    mutant_candidate_list = ['word_shuffling_candidate_mutants.csv', 'character_deleting_candidate_mutants.csv',
                        'symbol_insertion_candidate_mutants.csv', 'glyph_replacement_candidate_mutants.csv',
                        'character_splitting_candidate_mutants.csv', 'homophone_replacement_candidate_mutants.csv',
                        'nasal_replacement_candidate_mutants.csv', 'dorsal_replacement_candidate_mutants.csv',
                        'context_prediction_candidate_mutants.csv', 'synonym_replacement_candidate_mutants.csv',
                        'traditional_conversion_candidate_mutants.csv']
    final_mutants_list = ['word_shuffling_mutants.csv', 'character_deleting_mutants.csv',
                        'symbol_insertion_mutants.csv', 'glyph_replacement_mutants.csv',
                        'character_splitting_mutants.csv', 'homophone_replacement_mutants.csv',
                        'nasal_replacement_mutants.csv', 'dorsal_replacement_mutants.csv',
                        'context_prediction_mutants.csv', 'synonym_replacement_mutants.csv',
                        'traditional_conversion_mutants.csv']
    label_file_list = ['word_shuffling_labels.npy', 'character_deleting_labels.npy',
                       'symbol_insertion_labels.npy', 'glyph_replacement_labels.npy',
                       'character_splitting_labels.npy', 'homophone_replacement_labels.npy',
                       'nasal_replacement_labels.npy', 'dorsal_replacement_labels.npy',
                       'context_prediction_labels.npy', 'synonym_replacement_labels.npy',
                       'traditional_conversion_labels.npy']
    mut_name_list = ['word_shuffling', 'character_deleting',
                       'symbol_insertion', 'glyph_replacement',
                       'character_splitting', 'homophone_replacement',
                       'nasal_replacement', 'dorsal_replacement',
                       'context_prediction', 'synonym_replacement',
                       'traditional_conversion']
    test_data_path = os.path.join(source_data_path, 'source_data', 'test.csv')
    test_data = pd.read_csv(test_data_path)

    # labels = cal_output_label(test_data, model_path)
    # np.save(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'), labels)


    # for rate in rates:
    #     cal_candidate_output(model_path, source_data_path, rate)
    test_data_label = np.load(os.path.join(source_data_path, 'source_data', 'predict_lables.npy'))
    for name_num in range(len(mutant_candidate_list)):
        print(mutant_candidate_list[name_num])
        belong_list = []
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
        pre_list = []
        mut_list = []


        candidates_1 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[0]), mutant_candidate_list[name_num]))
        pre_label_1 = np.load(os.path.join(source_data_path, 'source_data', str(rates[0]), label_file_list[name_num]))

        candidates_2 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[1]), mutant_candidate_list[name_num]))
        pre_label_2 = np.load(os.path.join(source_data_path, 'source_data', str(rates[1]), label_file_list[name_num]))

        candidates_3 = pd.read_csv(os.path.join(source_data_path, 'source_data', str(rates[2]), mutant_candidate_list[name_num]))
        pre_label_3 = np.load(os.path.join(source_data_path, 'source_data', str(rates[2]), label_file_list[name_num]))

        for data_num in tqdm(range(len(test_data_label))):
            if int(test_data_label[data_num]) != int(pre_label_1[data_num]):
                belong_list.append(data_num)
                ori_text_list.append(candidates_1.loc[data_num, 'ori_text'])
                text_list.append(candidates_1.loc[data_num, 'text'])
                question_list.append(candidates_1.loc[data_num, 'question'])
                candidate_0_list.append(candidates_1.loc[data_num, 'candidate_0'])
                candidate_1_list.append(candidates_1.loc[data_num, 'candidate_1'])
                candidate_2_list.append(candidates_1.loc[data_num, 'candidate_2'])
                candidate_3_list.append(candidates_1.loc[data_num, 'candidate_3'])
                answer_list.append(candidates_1.loc[data_num, 'answer'])
                str_list.append(candidates_1.loc[data_num, 'str'])
                word_list.append(candidates_1.loc[data_num, 'word'])
                pre_list.append(pre_label_1[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_2[data_num]):
                belong_list.append(data_num)
                ori_text_list.append(candidates_2.loc[data_num, 'ori_text'])
                text_list.append(candidates_2.loc[data_num, 'text'])
                question_list.append(candidates_2.loc[data_num, 'question'])
                candidate_0_list.append(candidates_2.loc[data_num, 'candidate_0'])
                candidate_1_list.append(candidates_2.loc[data_num, 'candidate_1'])
                candidate_2_list.append(candidates_2.loc[data_num, 'candidate_2'])
                candidate_3_list.append(candidates_2.loc[data_num, 'candidate_3'])
                answer_list.append(candidates_2.loc[data_num, 'answer'])
                str_list.append(candidates_2.loc[data_num, 'str'])
                word_list.append(candidates_2.loc[data_num, 'word'])
                pre_list.append(pre_label_2[data_num])
                mut_list.append(mut_name_list[name_num])
                continue
            if int(test_data_label[data_num]) != int(pre_label_3[data_num]):
                belong_list.append(data_num)
                ori_text_list.append(candidates_3.loc[data_num, 'ori_text'])
                text_list.append(candidates_3.loc[data_num, 'text'])
                question_list.append(candidates_3.loc[data_num, 'question'])
                candidate_0_list.append(candidates_3.loc[data_num, 'candidate_0'])
                candidate_1_list.append(candidates_3.loc[data_num, 'candidate_1'])
                candidate_2_list.append(candidates_3.loc[data_num, 'candidate_2'])
                candidate_3_list.append(candidates_3.loc[data_num, 'candidate_3'])
                answer_list.append(candidates_3.loc[data_num, 'answer'])
                str_list.append(candidates_3.loc[data_num, 'str'])
                word_list.append(candidates_3.loc[data_num, 'word'])
                pre_list.append(pre_label_3[data_num])
                mut_list.append(mut_name_list[name_num])
        merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'text': text_list, 'question': question_list,
                         'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list, 'candidate_2': candidate_2_list,
                         'candidate_3': candidate_3_list,
                         'answer': answer_list, 'str': str_list, 'word': word_list, 'pre': pre_list, 'mut': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(save_path, final_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
