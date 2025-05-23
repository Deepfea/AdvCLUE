import os
import torch
from torch import nn
import pandas as pd
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.OCNLI.bert_model import BertClassifier
from method.step1_adv_sample_generation.OCNLI.Data import SentencePairDataset
from method.step1_adv_sample_generation.OCNLI.load_token import load_token
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def dev_epoch(epoch, model, dev_loader):
    model.eval()
    dev_loss = 0.0
    prediction_list = []
    reference_list = []
    for idx, batch in enumerate(dev_loader):
        loss, prediction, output = model(batch)
        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].clone().numpy())
        prediction_list.extend(prediction)
        reference_list.extend(labels)
        dev_loss += loss.item()
    dev_loss /= len(dev_loader)
    avg_accu = accuracy_score(reference_list, prediction_list) *100
    avg_accu = '{:.2f}'.format(avg_accu)
    dev_loss = '{:.4f}'.format(dev_loss)

    return dev_loss, avg_accu


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    eval_steps = 500
    lr = 1e-6
    epochs = 10
    batch_size = 32
    max_length = 256
    output_model_dir = '/media/usr/external/home/usr/project/project3_data/adv_samples/OCNLI/bert_base_chinese'
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/OCNLI'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def train(model, train_loader, dev_loader, config):
    epochs = config.epochs
    learning_rate = config.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_accu = 0
    step_cnt = 0
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        prediction_list = []
        reference_list = []
        for idx, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            loss, prediction, output = model(batch)
            loss.backward()
            optimizer.step()
            prediction = prediction.cpu().clone().numpy()
            labels = batch['labels'].numpy()
            prediction_list.extend(prediction)
            reference_list.extend(labels)
            train_loss += loss.item()
            step_cnt += 1
            # print(step_cnt)
            if step_cnt % config.eval_steps == 0:
                model.eval()
                dev_loss, dev_accu = dev_epoch(epoch=epoch, model=model, dev_loader=dev_loader)
                print("Training ----> Epoch: {}/{},  Batch: {}/{}*{}".format(epoch, epochs, idx + 1, epochs,
                                                                             len(train_loader)))
                print("Dev----------> loss={}, Accuracy={}".format(dev_loss, dev_accu))

                if float(dev_accu) > float(best_accu):
                    best_accu = dev_accu
                    torch.save(model.state_dict(),
                               os.path.join(config.output_model_dir, 'best_OCNLI_bert_base_chinese.model'))
                    # final_model_save_path = os.path.join(config.output_model_dir, 'best_LCQMC_bert_base_chinese.pt')
                    # torch.save(model, final_model_save_path)
                    print('new epoch saved as the best model {}'.format(epoch))
                model.train()
        train_loss /= len(train_loader)
        avg_accu = accuracy_score(reference_list, prediction_list) * 100
        avg_accu = '{:.2f}'.format(avg_accu)
        train_loss = '{:.4f}'.format(train_loss)
        print("-> loss={}, Accuracy={}".format(train_loss, avg_accu))
        model.eval()
        dev_loss, dev_accu = dev_epoch(epoch=epoch, model=model, dev_loader=dev_loader)
        print("* Dev epoch {}".format(epoch + 1))
        print("-> loss={}, Accuracy={}".format(dev_loss, dev_accu))

        if float(dev_accu) > float(best_accu):
            best_accu = dev_accu
            torch.save(model.state_dict(), os.path.join(config.output_model_dir, 'best_OCNLI_bert_base_chinese.model'))
            print('new epoch saved as the best model {}'.format(epoch))
        model.train()


def train_model(data_path):
    config = Config()
    model = BertClassifier(config)

    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_json = load_token(train_df)
    train_data = SentencePairDataset(train_json, config.max_length, padding_idx=0)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    dev_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    dev_json = load_token(dev_df)
    dev_data = SentencePairDataset(dev_json, config.max_length, padding_idx=0)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config.batch_size)
    train(model, train_loader, dev_loader, config)

if __name__ == "__main__":
    pass

