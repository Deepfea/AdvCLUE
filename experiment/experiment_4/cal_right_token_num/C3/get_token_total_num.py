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
    batch_size = 1
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
    results = []

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


        outputs = np.argmax(logits, axis=1)
        labels = label_ids.reshape(-1)
        temp_result = (outputs == labels).tolist()
        # print(temp_result)
        results.extend(temp_result)

    return results

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()
    config.load_path = os.path.join(base_tokenizer_path, model_name)
    config.BERT_PATH = os.path.join(base_model_path, dataset_name, model_name)

    tokenizer = BertTokenizer.from_pretrained(config.load_path)

    df_test = load_adv_data(dataset_name, base_adv_path)
    test_examples = create_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)

    get_token(test_dataset)


def get_token(test_dataset):
    token_list = []
    for num in range(len(test_dataset)):
        temp_list_0 = test_dataset[num][0][0]
        token_list.extend(temp_list_0)
        temp_list_1 = test_dataset[num][0][1]
        token_list.extend(temp_list_1)
        temp_list_2 = test_dataset[num][0][2]
        token_list.extend(temp_list_2)
        temp_list_3 = test_dataset[num][0][3]
        token_list.extend(temp_list_3)
    token_arr = np.unique(np.array(token_list))
    print(len(token_arr))


def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    return data

if __name__ == '__main__':

    pass
