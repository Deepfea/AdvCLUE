import os
import pandas as pd
from bert_score import score
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset
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

label_num = 15
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, label_num)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def cal_output_label(data, base_model_path, tokenizer_path):
    model_path = os.path.join(base_model_path, 'best_TNEWS_roberta_base_chinese.pt')
    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    dataset = MyDataset(data, tokenizer)
    test_loader = DataLoader(dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            temp_len = len(test_label)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if temp_len == 1:
                temp_list = []
                temp_list.append(temp_npy)
                temp_npy = np.array(temp_list)
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    predict_label = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_label = np.argmax(temp_npy)
        predict_label.append(temp_label)
    # predict_label = np.array(predict_label)
    # print(len(predict_label))
    return predict_label

def get_transferability_mutants(load_path):
    model_path = os.path.join(load_path, 'roberta_base_chinese')
    tokenizer_path = '/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese'
    for name_num in range(len(fidelity_mutants_list)):
        print(os.path.join(load_path, fidelity_mutants_list[name_num]))
        candidate_mutants = pd.read_csv(os.path.join(load_path, fidelity_mutants_list[name_num]))
        pre_labels = cal_output_label(candidate_mutants, model_path, tokenizer_path)
        true_labels = list(candidate_mutants['label'])
        # print(len(pre_labels))
        # print(len(true_labels))
        belong_list = []
        fact_list = []
        mutant_list = []
        str_list = []
        word_list = []
        label_list = []
        # pre_list = []
        mut_list = []
        for data_num in range(len(true_labels)):
            if int(pre_labels[data_num]) == int(true_labels[data_num]):
                continue
            belong_list.append(candidate_mutants.loc[data_num, 'belong'])
            fact_list.append(candidate_mutants.loc[data_num, 'ori_text'])
            mutant_list.append(candidate_mutants.loc[data_num, 'text'])
            str_list.append(candidate_mutants.loc[data_num, 'str'])
            word_list.append(candidate_mutants.loc[data_num, 'word'])
            label_list.append(candidate_mutants.loc[data_num, 'label'])
            # pre_list.append(candidate_mutants.loc[data_num, 'prediction'])
            mut_list.append(candidate_mutants.loc[data_num, 'type'])
        merge_dt_dict = {'belong': belong_list, 'ori_text': fact_list, 'str': str_list, 'word': word_list, 'text': mutant_list,
                                 'label': label_list, 'type': mut_list}
        data_df = pd.DataFrame(merge_dt_dict)
        data_df.to_csv(os.path.join(load_path, transferability_mutants_list[name_num]), index=False)
        print(len(data_df))

if __name__ == '__main__':
    pass
