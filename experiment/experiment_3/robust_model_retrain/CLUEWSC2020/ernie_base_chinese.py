import os
import pandas as pd
import torch
from transformers import BertTokenizer
from experiment.experiment_2.ori_model_train.CLUEWSC2020.dataset import get_dataloader
from experiment.experiment_2.ori_model_train.CLUEWSC2020.model import BERT

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    lr = 0.00001
    epochs = 5
    batch_size = 25
    max_length = 200
    output_model_dir = ''
    pretrained_model = ''

def train(dataset_path, model_name, pre_path, adv_path, save_path):
    config = Config()
    config.pretrained_model = os.path.join(pre_path, model_name)
    config.output_model_dir = os.path.join(save_path, model_name)
    model = BERT(config)
    model.to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    adv_df = pd.read_csv(os.path.join(adv_path, 'final_mutants.csv'))
    train_df = pd.concat([train_df, adv_df], ignore_index=True)

    train_dataloader = get_dataloader(train_df, tokenizer, config.max_length, config.batch_size)

    dev_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    dev_dataloader = get_dataloader(dev_df, tokenizer, config.max_length, config.batch_size)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    dev_acc = 0
    for epoch in range(config.epochs):
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
            for _, data in enumerate(dev_dataloader, 0):
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
            if not os.path.exists(config.output_model_dir):
                os.makedirs(config.output_model_dir)
            print(f"Validation Accuracy is updated: {epoch_accu}")

            model.embeddings.save_pretrained(config.output_model_dir)
            dev_acc = epoch_accu

if __name__ == "__main__":
    pass
