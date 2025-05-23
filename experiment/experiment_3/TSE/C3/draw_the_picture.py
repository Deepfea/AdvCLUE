import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from experiment.experiment_2.ori_model_train.C3.create_dataset import create_examples, get_dataset
from experiment.experiment_2.ori_model_train.C3.model import BertForClassification
import torch
from transformers import BertTokenizer
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE

class Config:
    max_length = 512
    epochs = 8
    batch_size = 4
    n_class = 4
    load_path = ''
    device = os.environ.get("DEVICE", "cuda:0")

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(config, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=eval_sampler)
    logits_all = []
    label_list = []

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
        temp_labels = list(label_ids.reshape(-1))
        label_list.extend(temp_labels)
        # print(logits_all)
        # print(label_list)

    output_arr = []
    for num in range(len(logits_all)):
        temp_output = logits_all[num]
        x = softmax(temp_output)
        output_arr.append(x)
    output_arr = np.array(output_arr)

    return output_arr, label_list

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):
    config = Config()
    config.load_path = os.path.join(base_tokenizer_path, model_name)
    tokenizer = BertTokenizer.from_pretrained(config.load_path)

    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    test_examples = create_examples(adv_data)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)

    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(base_model_path, 'ori_model', dataset_name, model_name, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    output_arr1, label_list1 = evaluate(config, model, test_dataset)

    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(base_model_path, 'retrained_model', dataset_name, model_name, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    output_arr2, label_list2 = evaluate(config, model, test_dataset)

    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

if __name__ == "__main__":
    pass

