import os
import json
import random

import pandas as pd
from tqdm import tqdm, trange

import numpy as np
import torch
# from apex import amp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.models.bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from create_dataset import create_examples, get_dataset
from model import BertForClassification

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def train(config, model, train_dataset, eval_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs

    optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup_steps,
                                                num_training_steps=t_total)

    # if config["fp16"] == 1:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    nb_tr_examples, nb_tr_steps = 0, 0
    model.zero_grad()
    train_iterator = trange(int(config.num_warmup_steps), desc="Epoch", disable=True)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.cuda() for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            tr_loss += loss.item()
            logging_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                stat = 'epoch {} | step {} | lr {:.6f} | loss {:.6f}'.format(epoch, global_step,
                                                                             scheduler.get_last_lr()[0], logging_loss)
                epoch_iterator.set_postfix_str(str(stat))
                logging_loss = 0.0

        # Save model checkpoint
        eval_loss, eval_metric, eval_logits = evaluate(config, model, eval_dataset)
        print("epoch: {}, eval_result: {:.6f}, eval_loss: {:.6f}".format(epoch, eval_metric, eval_loss))
        save_dir = os.path.join(config.save_dir, 'checkpoint-{}'.format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'model.bin'))
        torch.save(config, os.path.join(save_dir, 'training_args.bin'))
        print("Saving model checkpoint to {}".format(save_dir))

    return global_step, tr_loss / global_step


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

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy, logits_all

class Config:
    max_length = 512
    epochs = 5
    batch_size = 4
    lr = 2e-5
    gradient_accumulation_steps = 6
    num_warmup_steps = 500
    max_grad_norm = 1.0
    n_class = 4
    fp16 = 2
    save_dir = ''
    load_path = ''
    device = os.environ.get("DEVICE", "cuda:0")

def train_model(dataset_path, save_path, load_path, model_name):
    config = Config()
    config.save_dir = os.path.join(save_path, model_name)
    config.load_path = os.path.join(load_path, model_name)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    tokenizer = BertTokenizer.from_pretrained(config.load_path)
    model = BertForClassification(config.load_path)
    model.to(config.device)

    df_train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_examples = create_examples(df_train)
    train_dataset = get_dataset(train_examples, tokenizer, config.max_length)

    df_test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_examples = create_examples(df_test)
    test_dataset = get_dataset(test_examples, tokenizer, config.max_length)

    train(config, model, train_dataset, test_dataset)



if __name__ == '__main__':
    pass


