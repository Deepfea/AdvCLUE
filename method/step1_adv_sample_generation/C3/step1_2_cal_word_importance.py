import numpy as np
import os
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
from method.step1_adv_sample_generation.C3.seg_data_to_words import seg_data
from torch.utils.data import DataLoader
from create_dataset import create_examples, get_dataset
from transformers import BertTokenizer
from tqdm import tqdm
from model import BertForClassification


def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def gen_new_sentence(data_path):
    belong_list = []
    str_list = []
    fact_list = []
    question_list = []
    candidate_0_list = []
    candidate_1_list = []
    candidate_2_list = []
    candidate_3_list = []
    answer_list = []

    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·', '\n']
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)

    for data_num in range(len(seg_arr)):
        temp_segs = seg_arr[data_num]
        fact = test_data.loc[data_num, 'text']
        question = test_data.loc[data_num, 'question']
        candidate_0 = test_data.loc[data_num, 'candidate_0']
        candidate_1 = test_data.loc[data_num, 'candidate_1']
        candidate_2 = test_data.loc[data_num, 'candidate_2']
        candidate_3 = test_data.loc[data_num, 'candidate_3']
        answer = test_data.loc[data_num, 'answer']

        for seg_num in range(len(temp_segs)):
            temp_seg = temp_segs[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            temp_fact = fact.replace(temp_seg, '')
            belong_list.append(data_num)
            str_list.append(temp_seg)
            fact_list.append(temp_fact)
            question_list.append(question)
            candidate_0_list.append(candidate_0)
            candidate_1_list.append(candidate_1)
            candidate_2_list.append(candidate_2)
            candidate_3_list.append(candidate_3)
            answer_list.append(answer)

    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'text': fact_list, 'question': question_list,
                     'candidate_0': candidate_0_list, 'candidate_1': candidate_1_list, 'candidate_2': candidate_2_list,
                     'candidate_3': candidate_3_list, 'answer': answer_list}
    data_df = pd.DataFrame(merge_dt_dict)

    return data_df

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

def get_label(test_dataset):
    labels = []
    for num in range(len(test_dataset)):
        temp_label = test_dataset[num][3]
        labels.append(int(temp_label))
    return labels

def cal_word_importance(save_path, base_model_path):
    config = Config()
    data_path = os.path.join(save_path, 'source_data')
    segmentation(data_path)
    new_data_df = gen_new_sentence(data_path)
    print(len(new_data_df))
    tokenizer = BertTokenizer.from_pretrained(config.load_path)
    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(base_model_path, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    test_examples = create_examples(new_data_df)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    labels = get_label(test_dataset)
    print(len(labels))
    output_list = evaluate(config, model, test_dataset)
    output_arr = []
    for num in range(len(output_list)):
        temp_output = output_list[num]
        x = softmax(temp_output)
        output_arr.append(x)
    output_arr = np.array(output_arr)

    source_data_output = np.load(os.path.join(data_path, 'testing_data_output.npy'))

    importance_value = []
    for data_num in range(len(output_arr)):
        temp_output = np.array(output_arr[data_num])
        temp_label = labels[data_num]
        temp_value = source_data_output[new_data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    new_data_df['importance_value'] = importance_value
    print(len(new_data_df))
    df_filtered = new_data_df[new_data_df['importance_value'] > -100].reset_index(drop=True)
    print(len(df_filtered))
    df_filtered.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)

    seg_list = []
    seg_score = []
    for class_num in range(len(source_data_output)):
        seg_list.append([])
        seg_score.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
    print(seg_list)
    print(seg_score)

    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)

    np.save(os.path.join(data_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(data_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))

if __name__ == '__main__':
    pass
