import json
import os

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from bert_model import BertClassifier
from Data import *
from transformers import AdamW
from sklearn.metrics import accuracy_score
from load_token import load_token

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
    eval_steps = 250
    lr = 1e-6
    epochs = 5
    batch_size = 16
    max_length = 510
    output_model_dir = ''
    pretrained_model = ''

def train(model, train_loader, dev_loader, config, model_name):
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
                print("Training ----> Epoch: {}/{},  Batch: {}/{}*{}".format(epoch, epochs, idx + 1, epochs, len(train_loader)))
                print("Dev----------> loss={}, Accuracy={}".format(dev_loss, dev_accu))

                if float(dev_accu) > float(best_accu):
                    best_accu = dev_accu
                    torch.save(model.state_dict(),
                               os.path.join(config.output_model_dir, 'best_CSL_' + model_name + '.model'))
                    print('new epoch saved as the best model {}'.format(epoch))
                model.train()
        train_loss /= len(train_loader)
        avg_accu = accuracy_score(reference_list, prediction_list) *100
        avg_accu = '{:.2f}'.format(avg_accu)
        train_loss = '{:.4f}'.format(train_loss)
        print("-> loss={}, Accuracy={}".format(train_loss, avg_accu))
        model.eval()
        dev_loss, dev_accu = dev_epoch(epoch=epoch, model=model, dev_loader=dev_loader)
        print("* Dev epoch {}".format(epoch + 1))
        print("-> loss={}, Accuracy={}".format(dev_loss, dev_accu))

        if float(dev_accu) > float(best_accu):
            best_accu = dev_accu
            torch.save(model.state_dict(), os.path.join(config.output_model_dir, 'best_CSL_' + model_name + '.model'))
            print('new epoch saved as the best model {}'.format(epoch))
        model.train()

def test(model, test_loader, config):
    test_loss, avg_accu = dev_epoch(epoch=0, config=config, model=model, dev_loader=test_loader)
    print("* Result on test set")
    print("-> loss={}, Accuracy={}, Precision={}, Recall={}, F1={}".format(test_loss, avg_accu))

def train_model(data_path, save_path, load_path, model_name):
    config = Config()
    config.pretrained_model = os.path.join(load_path, model_name)
    config.output_model_dir = os.path.join(save_path, model_name)
    if not os.path.exists(config.output_model_dir):
        os.makedirs(config.output_model_dir)

    model = BertClassifier(config)
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_json = load_token(train_df, config.pretrained_model)
    train_data = SentencePairDataset(train_json, config.max_length, padding_idx=0)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    dev_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    dev_json = load_token(dev_df, config.pretrained_model)
    dev_data = SentencePairDataset(dev_json, config.max_length, padding_idx=0)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config.batch_size)

    train(model, train_loader, dev_loader, config, model_name)

if __name__ == "__main__":
    pass








