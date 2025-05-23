from torch.optim import Adam
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch import nn

from method.step1_adv_sample_generation.TNEWS.create_dataset import MyDataset

model_name = 'bert_base_chinese'
dataset_name = 'TNEWS'
load_path = os.path.join('/media/usr/external/home/usr/project/project3_data/base_model', model_name)

label_num = 15

tokenizer = BertTokenizer.from_pretrained(load_path)
model = BertModel.from_pretrained(load_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, label_num)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def evaluate(model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(dataset): .4f}')

def main():
    save_path = os.path.join('/media/usr/external/home/usr/project/project3_data/experiment/experiment2/AdvCLUE/ori_model/TNEWS')
    dataset_path = '/media/usr/external/home/usr/project/project3_data/dataset/TNEWS'
    print("加载数据集......")
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_dataset = MyDataset(train_data, tokenizer)
    val_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    val_dataset = MyDataset(val_data, tokenizer)
    print("加载数据集完成")

    epoch = 5
    batch_size = 32
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 20241118
    setup_seed(random_seed)

    # 定义模型
    model = BertClassifier()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = criterion.to(device)

    # 构建数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 训练
    best_dev_acc = 0
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for inputs, labels in dev_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()

            print(f'''Epochs: {epoch_num + 1}
              | Train Loss: {total_loss_train / len(train_dataset): .3f}
              | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
              | Val Loss: {total_loss_val / len(val_dataset): .3f}
              | Val Accuracy: {total_acc_val / len(val_dataset): .3f}''')

            # 保存最优的模型
            if total_acc_val / len(val_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(val_dataset)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model_save_path = os.path.join(save_path, model_name)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                final_model_save_path = os.path.join(model_save_path, 'best_' + dataset_name + '_' + model_name + '.pt')
                torch.save(model, final_model_save_path)

        model.train()


if __name__ == '__main__':
    pass




