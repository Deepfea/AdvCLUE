import numpy as np
import pandas as pd
from experiment.experiment_2.ori_model_train.CMRC2018.load_token import get_token
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from experiment.experiment_2.ori_model_train.CMRC2018.create_dataset import SquadDataset
import torch

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = ''
    pretrained_model = ''

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def del_data(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    load_model_path = os.path.join(base_model_path, dataset_name, model_name)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)

    test_df = load_adv_data(dataset_name, base_adv_path)
    test_encodings = get_token(test_df, config.pretrained_model)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    start_flag_list, end_flag_list = get_acc(model, test_loader)

    del_num = int(len(start_flag_list) * 0.5)

    belong_list = []
    ori_context_list = []
    ori_question_list = []
    ori_answer_start_list = []
    ori_answer_end_list = []
    str_list = []
    word_list = []
    answer_list = []
    context_list = []
    question_list = []
    answer_start_list = []
    answer_end_list = []
    type_list = []
    error_num_list = []
    temp_num = 0

    for num in range(len(start_flag_list)):
        if (start_flag_list[num] == 0 or end_flag_list[num] == 0) and temp_num < del_num:
            temp_num += 1
            continue
        belong_list.append(test_df.loc[num, 'belong'])
        ori_context_list.append(test_df.loc[num, 'ori_context'])
        ori_question_list.append(test_df.loc[num, 'ori_question'])
        ori_answer_start_list.append(test_df.loc[num, 'ori_answer_start'])
        ori_answer_end_list.append(test_df.loc[num, 'ori_answer_end'])
        str_list.append(test_df.loc[num, 'str'])
        word_list.append(test_df.loc[num, 'word'])
        context_list.append(test_df.loc[num, 'context'])
        question_list.append(test_df.loc[num, 'question'])
        answer_start_list.append(test_df.loc[num, 'answer_start'])
        answer_end_list.append(test_df.loc[num, 'answer_end'])
        answer_list.append(test_df.loc[num, 'answer'])
        type_list.append(test_df.loc[num, 'type'])
        error_num_list.append(test_df.loc[num, 'error_num'])
    merge_dt_dict = {'belong': belong_list, 'ori_context': ori_context_list, 'ori_question': ori_question_list,
                     'ori_answer_start': ori_answer_start_list, 'ori_answer_end': ori_answer_end_list,
                     'str': str_list, 'word': word_list,
                     'context': context_list, 'question': question_list, 'answer_start': answer_start_list,
                     'answer_end': answer_end_list, 'answer': answer_list, 'type': type_list,
                     'error_num': error_num_list}

    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))
    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)


def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    train_data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    return train_data

def get_acc(model, dev_loader):
    model.eval()
    start_flag_list = []
    end_flag_list = []
    for idx, batch in enumerate(dev_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)

        acc_start = (start_pred == start_positions).cpu().numpy()
        acc_end = (end_pred == end_positions).cpu().numpy()

        start_flag_list += list(acc_start)
        end_flag_list += list(acc_end)

    return start_flag_list, end_flag_list

if __name__ == '__main__':
    pass
