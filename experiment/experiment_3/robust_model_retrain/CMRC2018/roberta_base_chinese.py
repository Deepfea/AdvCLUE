import pandas as pd
from tqdm.autonotebook import tqdm
import torch
import os
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from experiment.experiment_2.ori_model_train.CMRC2018.load_token import get_token
from experiment.experiment_2.ori_model_train.CMRC2018.create_dataset import SquadDataset


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    eval_steps = 500
    lr = 5e-5
    epochs = 5
    eval_steps = 200
    batch_size = 16
    max_length = 510
    output_model_dir = ''
    pretrained_model = ''

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

def train(train_loader, test_loader, model, config, model_name):

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
                               os.path.join(config.output_model_dir, 'best_CMRC2018_' + model_name + '.model'))
                model.train()

        model.eval()
        dev_acc_start_sum, dev_acc_end_sum = dev_epoch(model, test_loader)
        if float(best_acc_start_sum) < float(dev_acc_start_sum) and float(best_acc_end_sum) < float(dev_acc_end_sum):
            best_acc_start_sum = dev_acc_start_sum
            best_acc_end_sum = dev_acc_end_sum
            torch.save(model.state_dict(),
                       os.path.join(config.output_model_dir, 'best_CMRC2018_' + model_name + '.model'))
        model.train()

def train_model(dataset_path, load_path, save_path, adv_path, model_name):
    config = Config()
    config.pretrained_model = os.path.join(load_path, model_name)
    config.output_model_dir = os.path.join(save_path, model_name)
    if not os.path.exists(config.output_model_dir):
        os.makedirs(config.output_model_dir)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    model = model.to(config.device)
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    adv_data = pd.read_csv(os.path.join(adv_path, 'final_mutants.csv'))
    train_data = pd.concat([train_data, adv_data], ignore_index=True)

    train_encodings = get_token(train_data, config.pretrained_model)
    train_dataset = SquadDataset(train_encodings, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_encodings = get_token(test_data, config.pretrained_model)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    train(train_loader, test_loader, model, config, model_name)

if __name__ == "__main__":
    pass


