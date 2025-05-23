import os

import pandas as pd
import torch
from transformers import BertTokenizer
from dataset import get_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    lr = 0.00001
    epochs = 15
    batch_size = 25
    max_length = 200
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese"

def get_wsc_dataloader(max_len=200, batch_size=10):
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    train_df = pd.read_csv(os.path.join('/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020/train.csv'))
    train_dataloader = get_dataloader(train_df, tokenizer, max_len, batch_size)

    dev_df = pd.read_csv(os.path.join('/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020/test.csv'))
    dev_dataloader = get_dataloader(dev_df, tokenizer, max_len, batch_size)
    return train_dataloader, dev_dataloader

def train(save_path):
    config = Config()
    model = BERT(config)
    model.to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    train_dataloader, eval_dataloader = get_wsc_dataloader(
        config.max_length, config.batch_size
    )
    dev_acc = 0
    for epoch in range(config.epochs):
        print(epoch)
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for _, data in enumerate(train_dataloader, 0):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(
                config.device, dtype=torch.long
            )
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (
                input_ids,
                attention_mask,
                span1_begin,
                span1_end,
                span2_begin,
                span2_end,
            )
            outputs = model(input)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}"
        )
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")

        model.eval()
        n_correct = 0
        nb_tr_examples = 0
        with torch.no_grad():
            for _, data in enumerate(eval_dataloader, 0):
                input_ids = data["input_ids"].to(config.device, dtype=torch.long)
                attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
                targets = data["labels"].to(config.device, dtype=torch.long)
                span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
                span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
                span1_end = data["span1_end"].to(config.device, dtype=torch.long)
                span2_end = data["span2_end"].to(config.device, dtype=torch.long)
                input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
                outputs = model(input)
                big_val, big_idx = torch.max(outputs, dim=1)
                n_correct += calcuate_accu(big_idx, targets)
                nb_tr_examples += targets.size(0)
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Validation Accuracy Epoch: {epoch_accu}")
        if epoch_accu > dev_acc:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Validation Accuracy is updated: {epoch_accu}")

            model.embeddings.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            dev_acc = epoch_accu



if __name__ == "__main__":
    pass
