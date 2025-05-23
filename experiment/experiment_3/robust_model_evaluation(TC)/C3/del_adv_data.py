import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from experiment.experiment_2.ori_model_train.C3.create_dataset import create_examples, get_dataset
from experiment.experiment_2.ori_model_train.C3.model import BertForClassification
from transformers import BertTokenizer

class Config:
    max_length = 512
    epochs = 8
    batch_size = 4
    n_class = 4
    load_path = ''
    BERT_PATH = ''
    device = os.environ.get("DEVICE", "cuda:0")

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return outputs == labels

def get_acc(model, eval_dataset):
    config = Config()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=eval_sampler)

    flag_list = []

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
        temp_list = accuracy(logits, label_ids.reshape(-1))
        flag_list.extend(temp_list)
    return flag_list

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.load_path = os.path.join(base_tokenizer_path, model_name)
    config.BERT_PATH = os.path.join(base_model_path, dataset_name, model_name)

    tokenizer = BertTokenizer.from_pretrained(config.load_path)
    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(config.BERT_PATH, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

    df_test = load_adv_data(dataset_name, base_adv_path)
    test_examples = create_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    flag_list = get_acc(model, test_dataset)
    del_num = int(len(flag_list) * 0.5)

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
    type_list = []
    error_num_list = []
    temp_num = 0
    for num in range(len(flag_list)):
        if flag_list[num] == 0 and temp_num < del_num:
            temp_num += 1
        belong_list.append(df_test.loc[num, 'belong'])
        ori_text_list.append(df_test.loc[num, 'ori_text'])
        str_list.append(df_test.loc[num, 'str'])
        word_list.append(df_test.loc[num, 'word'])
        text_list.append(df_test.loc[num, 'text'])
        question_list.append(df_test.loc[num, 'question'])
        candidate_0_list.append(df_test.loc[num, 'candidate_0'])
        candidate_1_list.append(df_test.loc[num, 'candidate_1'])
        candidate_2_list.append(df_test.loc[num, 'candidate_2'])
        candidate_3_list.append(df_test.loc[num, 'candidate_3'])
        answer_list.append(df_test.loc[num, 'answer'])
        label_list.append(df_test.loc[num, 'label'])
        type_list.append(df_test.loc[num, 'type'])
        error_num_list.append(df_test.loc[num, 'error_num'])

    merge_dt_dict = {'belong': belong_list, 'ori_text': ori_text_list, 'str': str_list, 'word': word_list,
                     'text': text_list, 'question': question_list,
                     'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list, 'candidate_2': candidate_2_list,
                     'candidate_3': candidate_3_list, 'answer': answer_list, 'label': label_list, 'type': type_list,
                     'error_num': error_num_list
                     }

    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))
    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    return data

if __name__ == '__main__':
    pass