import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from method.step1_adv_sample_generation.CHID.create_dataset import ClozeDataset
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
from method.step1_adv_sample_generation.CHID.utils import EarlyStopping
import json
import re
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')

import transformers
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup # WarmupLinearSchedule
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold

workdir = os.getcwd()
project_dir = os.path.split(workdir)[0]
data_dir = os.path.join(project_dir, "data")
midata_dir = os.path.join(data_dir, "midata")


class Config:
    max_len = 510
    batch_size = 8
    epochs = 5
    lr = 5e-5
    device = os.environ.get("DEVICE", "cuda:0")
    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese'
    # BERT_PATH = project_dir + "/pretrained_models/ernie_based_pretrained"
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)
    output_model_dir = '/media/usr/external/home/usr/project/project3_data/adv_samples/CHID/bert_base_chinese'


def convert_to_features(df_data, candidates, config):
    dataset = ClozeDataset(tokenizer=config.TOKENIZER, data_id=df_data.data_id.values, tag=df_data.tag.values,
                           text=df_data.text.values, candidate=candidates,
                           groundTruth=df_data.groundTruth.values, max_len=config.max_len)
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


def train_model(dataset_path):
    idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
    idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
    config = Config()
    df_test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))[:100]
    train_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    train_dataset = convert_to_features(df_test, train_candidates, config)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=0)
    print(len(train_data_loader))
    print(len(train_dataset))

    df_dev = pd.read_csv(os.path.join(dataset_path, 'test.csv'))[:100]
    test_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    valid_dataset = convert_to_features(df_dev, test_candidates, config)
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
        if es.early_stop:
            print("********** Early stopping ********")
            break


def predict():
    df_test = read_data(os.path.join(data_dir, "test_data.txt"))
    test_dataset = convert_to_features(df_test, midata_dir + f"/test_features.pkl")
    # Instantiate DataLoader with `test_dataset`
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.TEST_BATCH_SIZE
    )
    # Load pretrained BERT (bert-base-uncased)
    model_path = config.MODEL_SAVE_PATH
    model_config = transformers.BertConfig.from_pretrained(model_path)
    model_path = config.MODEL_SAVE_PATH
    # Instantiate our model with `model_config`
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config, idiom_num=len(idiom_vocab))
    # # Load each of the five trained models and move to GPU
    trained_model_path = os.path.join(model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    pred_labels = []
    true_labels = []
    # Turn of gradient calculations
    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))
        # Predict the span containing the sentiment for each batch
        for bi, batch in enumerate(tk0):
            # Predict logits
            logits = run_one_step(batch, model, device)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=1).tolist()
            # pred_logits = np.append(pred_logits, logits.view(-1).cpu().detach().numpy())
            pred_labels.extend(pred_label)
            true_labels.extend(batch["label"].numpy())
    # return pred_logits
    test_acc = accuracy_score(true_labels, pred_labels)
    logger.info(f"test acc: {test_acc}")
    return pred_labels


if __name__ == '__main__':
    pass

