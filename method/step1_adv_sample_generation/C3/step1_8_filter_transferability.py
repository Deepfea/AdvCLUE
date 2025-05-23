import numpy as np
import os
from torch.utils.data import DataLoader, SequentialSampler
import torch
import pandas as pd
from transformers import BertTokenizer
from model import BertForClassification
from create_dataset import create_examples, get_dataset
from tqdm import tqdm

fidelity_mutants_list = ['fidelity_word_shuffling_mutants.csv', 'fidelity_character_deleting_mutants.csv',
                      'fidelity_symbol_insertion_mutants.csv', 'fidelity_glyph_replacement_mutants.csv',
                      'fidelity_character_splitting_mutants.csv', 'fidelity_homophone_replacement_mutants.csv',
                      'fidelity_nasal_replacement_mutants.csv', 'fidelity_dorsal_replacement_mutants.csv',
                      'fidelity_context_prediction_mutants.csv', 'fidelity_synonym_replacement_mutants.csv',
                      'fidelity_traditional_conversion_mutants.csv']

transferability_mutants_list = ['transferability_word_shuffling_mutants.csv', 'transferability_character_deleting_mutants.csv',
                      'transferability_symbol_insertion_mutants.csv', 'transferability_glyph_replacement_mutants.csv',
                      'transferability_character_splitting_mutants.csv', 'transferability_homophone_replacement_mutants.csv',
                      'transferability_nasal_replacement_mutants.csv', 'transferability_dorsal_replacement_mutants.csv',
                      'transferability_context_prediction_mutants.csv', 'transferability_synonym_replacement_mutants.csv',
                      'transferability_traditional_conversion_mutants.csv']

class Config:
    max_length = 512
    epochs = 8
    batch_size = 4
    n_class = 4
    load_path = '/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese'
    device = os.environ.get("DEVICE", "cuda:0")

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def get_label(test_dataset):
    labels = []
    for num in range(len(test_dataset)):
        temp_label = test_dataset[num][3]
        labels.append(int(temp_label))
    return labels

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
    true_label = get_label(test_dataset)
    output_list = evaluate(config, model, test_dataset)
    pre_labels = []
    for num in range(len(output_list)):
        temp_arr = output_list[num]
        pre_labels.append(np.argmax(temp_arr))
    output_arr = np.array(pre_labels)

    return output_arr, true_label

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese')
    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        pre_labels, true_labels = cal_output_label(candidate_mutants, model_path)
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
        label_list = []
        mut_list = []

        for data_num in range(len(true_labels)):
            if int(pre_labels[data_num]) == int(true_labels[data_num]):
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
            label_list.append(true_labels[data_num])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])
        merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list,
                         'word': word_list, 'text': text_list, 'question': question_list,
                         'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list,
                         'candidate_2': candidate_2_list, 'candidate_3': candidate_3_list,
                         'answer': answer_list, 'label': label_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
