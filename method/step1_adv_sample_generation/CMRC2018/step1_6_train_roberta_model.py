import pandas as pd
from tqdm.autonotebook import tqdm
import torch
import os
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from method.step1_adv_sample_generation.CMRC2018.load_token_roberta import get_token
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    eval_steps = 500
    lr = 5e-5
    epochs = 5
    eval_steps = 200
    batch_size = 16
    max_length = 510
    output_model_dir = '/media/usr/external/home/usr/project/project3_data/adv_samples/CMRC2018/final_adv/roberta_base_chinese'
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/roberta_base_chinese"

def dev_epoch(model, dev_loader):
    model.eval()
    acc_start_sum = 0.0
    acc_end_sum = 0.0
    for idx, batch in enumerate(dev_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)

        acc_start = (start_pred == start_positions).float().mean()
        acc_end = (end_pred == end_positions).float().mean()

        acc_start_sum += acc_start
        acc_end_sum += acc_end
    return acc_start_sum, acc_end_sum

def train(train_loader, test_loader, model, config):

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_acc_start_sum = 0
    best_acc_end_sum = 0
    model.train()
    step_cnt = 0
    for epoch in range(config.epochs):
        loss_sum = 0.0
        acc_start_sum = 0.0
        acc_end_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            optim.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss.mean()
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            start_pred = torch.argmax(outputs.start_logits, dim=1)
            end_pred = torch.argmax(outputs.end_logits, dim=1)

            acc_start = (start_pred == start_positions).float().mean()
            acc_end = (end_pred == end_positions).float().mean()

            acc_start_sum += acc_start
            acc_end_sum += acc_end

            # Update progress bar
            postfix = {
                "loss": f"{loss_sum / (batch_idx + 1):.4f}",
                "acc_start": f"{acc_start_sum / (batch_idx + 1):.4f}",
                "acc_end": f"{acc_end_sum / (batch_idx + 1):.4f}",
            }

            # Add batch accuracy to progress bar
            batch_desc = f"Epoch {epoch}, train loss: {postfix['loss']}"
            pbar.set_postfix_str(
                f"{batch_desc}, acc start: {postfix['acc_start']}, acc end: {postfix['acc_end']}"
            )
            step_cnt += 1
            if step_cnt % config.eval_steps == 0:
                model.eval()
                dev_acc_start_sum, dev_acc_end_sum = dev_epoch(model, test_loader)
                if float(best_acc_start_sum) < float(dev_acc_start_sum) and float(best_acc_end_sum) < float(
                        dev_acc_end_sum):
                    best_acc_start_sum = dev_acc_start_sum
                    best_acc_end_sum = dev_acc_end_sum
                    print('new epoch saved as the best model {}'.format(epoch))
                    torch.save(model.state_dict(),
                               os.path.join(config.output_model_dir, 'best_CMRC2018_roberta_base_chinese.model'))
                model.train()

        model.eval()
        dev_acc_start_sum, dev_acc_end_sum = dev_epoch(model, test_loader)
        if float(best_acc_start_sum) < float(dev_acc_start_sum) and float(best_acc_end_sum) < float(dev_acc_end_sum):
            best_acc_start_sum = dev_acc_start_sum
            best_acc_end_sum = dev_acc_end_sum
            torch.save(model.state_dict(),
                       os.path.join(config.output_model_dir, 'best_CMRC2018_roberta_base_chinese.model'))
        model.train()

def train_model(dataset_path):
    config = Config()
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    model = model.to(config.device)
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_encodings = get_token(train_data)
    # print(train_encodings)
    train_dataset = SquadDataset(train_encodings, config)
    # print(train_dataset.encodings)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_encodings = get_token(test_data)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    if not os.path.exists(config.output_model_dir):
        os.makedirs(config.output_model_dir)

    train(train_loader, test_loader, model, config)

if __name__ == "__main__":
    pass