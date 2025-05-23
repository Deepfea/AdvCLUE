import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from method.step1_adv_sample_generation.C3.create_dataset import create_examples, get_dataset
from method.step1_adv_sample_generation.C3.model import BertForClassification
from transformers import BertTokenizer

class Config:
    max_length = 512
    epochs = 8
    batch_size = 4
    n_class = 4
    load_path = '/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/C3/bert_base_chinese'
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

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_accuracy

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.load_path)
    model = BertForClassification(config.load_path)
    trained_model_path = os.path.join(base_model_path, "model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

    df_test = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_examples = create_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    acc1 = get_acc(model, test_dataset)

    df_test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_examples = create_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)
    acc2 = get_acc(model, test_dataset)

    print((acc1 + acc2) / 2.0)

if __name__ == '__main__':
    pass
