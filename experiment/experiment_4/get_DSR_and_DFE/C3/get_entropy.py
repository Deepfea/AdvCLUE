import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from experiment.experiment_2.ori_model_train.C3.create_dataset import create_examples, get_dataset, create_ori_examples
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
    return np.sum(outputs == labels)

def get_acc(model, eval_dataset):
    config = Config()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=eval_sampler)

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
            # print(outputs)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()

        for i in range(len(logits)):
            logits_all += [logits[i]]
    return logits_all

def get_entropy(outputs1, outputs2):
    results = []
    for num in range(len(outputs1)):
        temp_output1 = outputs1[num]
        temp_output2 = outputs2[num]
        dot_product = np.dot(temp_output1, temp_output2)
        norm_temp_output1 = np.linalg.norm(temp_output1)
        norm_temp_output2 = np.linalg.norm(temp_output2)
        similarity = dot_product / (norm_temp_output1 * norm_temp_output2)
        results.append(similarity)
    results = np.array(results)
    total_sum = np.sum(results)
    final_result = 0
    for result_num in range(len(results)):
        temp_result = - results[result_num] / total_sum * np.log2(results[result_num] / total_sum)
        if np.isnan(temp_result):
            temp_result = 1
        final_result += temp_result
    return final_result

def cal_entropy(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
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
    outputs1 = get_acc(model, test_dataset)

    df_test = load_adv_data(dataset_name, base_adv_path)
    test_examples = create_ori_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    outputs2 = get_acc(model, test_dataset)

    result = get_entropy(outputs1, outputs2)

    print(result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return data

if __name__ == '__main__':
    pass
