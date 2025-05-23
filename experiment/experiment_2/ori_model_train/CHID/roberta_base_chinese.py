import os
import torch
import pandas as pd
import numpy as np
from create_dataset import ClozeDataset
from bert_model import BertForClozeBaseline
from utils import EarlyStopping
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup # WarmupLinearSchedule
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold


class Config:
    batch_size = 32
    epochs = 5
    lr = 5e-5
    device = os.environ.get("DEVICE", "cuda:0")
    BERT_PATH = ''
    TOKENIZER = ''
    output_model_dir = ''


def convert_to_features(df_data, candidates, candidate_ids, path):
    dataset = ClozeDataset(df_data, candidates, candidate_ids, path)
    datas = []
    data = []
    batch_id = 1

    for bi, item in enumerate(tqdm(dataset, total=len(dataset))):
        data.append(item)
        if len(data) == 50000 or bi == len(dataset) - 1:
            batch_id += 1
            datas.extend(data)
            data = []
    dataset = datas

    return dataset


def run_one_step(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    input_mask = batch["input_masks"].to(device)
    position = batch["position"].to(device)
    idiom_ids = batch["idiom_ids"].to(device)


    logits = model(
        input_ids,
        input_mask,
        token_type_ids=token_type_ids,
        idiom_ids=idiom_ids,
        positions=position,
    )  #

    return logits


def train_fn(data_loader, model, optimizer, device, epoch, scheduler=None):

    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, batch in enumerate(tk0):
        model.zero_grad()
        logits = run_one_step(batch, model, device)
        label = batch["label"].to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, label)

        loss.backward()
        optimizer.step()   #更新模型参数
        optimizer.zero_grad()

        scheduler.step()

        outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        pred_label = np.argmax(outputs, axis=1)
        acc = accuracy_score(label.cpu().numpy(), pred_label)

        tk0.set_postfix(epoch=epoch, acc=acc, loss=loss.item())

def eval_fn(valid_data_loader, model, device):

    model.eval()
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))

        for bi, batch in enumerate(tk0):

            logits = run_one_step(batch, model, device)
            label = batch["label"].to(device)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, label)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=1)
            acc = accuracy_score(label.cpu().numpy(), pred_label)
            pred_labels.extend(pred_label.tolist())
            true_labels.extend(label.cpu().numpy().tolist())

            tk0.set_postfix(loss=loss.item(), acc=acc)

    total_acc = accuracy_score(true_labels, pred_labels)
    return total_acc

def train_model(dataset_path, save_path, load_path, model_name):
    idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
    idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
    config = Config()
    config.BERT_PATH = os.path.join(load_path, model_name)
    config.TOKENIZER = BertTokenizer.from_pretrained(config.BERT_PATH, lowercase=True)
    config.output_model_dir = os.path.join(save_path, model_name)
    if not os.path.exists(config.output_model_dir):
        os.makedirs(config.output_model_dir)

    df_train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_candidates = np.load(os.path.join(dataset_path, 'train_candidates.npy'), allow_pickle=True)
    train_candidate_ids = np.load(os.path.join(dataset_path, 'train_candidate_ids.npy'), allow_pickle=True)

    train_dataset = convert_to_features(df_train, train_candidates, train_candidate_ids, config.BERT_PATH)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=0)

    df_dev = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(dataset_path, 'test_candidate_ids.npy'),  allow_pickle=True)

    valid_dataset = convert_to_features(df_dev, test_candidates, test_candidate_ids, config.BERT_PATH)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=0)

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.BERT_PATH,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    model.to(config.device)

    train(train_data_loader, len(train_dataset), valid_data_loader, len(valid_dataset), config, model)


def train(train_data_loader, train_data_num, valid_data_loader, val_data_num, config, model):

    num_train_steps = int(train_data_num / config.batch_size * config.epochs)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_train_steps)

    es = EarlyStopping(patience=2, mode="max", delta=0.00001)

    for epoch in range(config.epochs):
        train_fn(train_data_loader, model, optimizer, config.device, epoch + 1, scheduler)
        eval_acc = eval_fn(valid_data_loader, model, config.device)
        print(f"epoch: {epoch + 1}, acc = {eval_acc}")
        es(epoch, eval_acc, model, model_path=config.output_model_dir)
        # if es.early_stop:
        #     print("********** Early stopping ********")
        #     break

if __name__ == '__main__':
    pass

