import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.PKU.create_PKU_dataset import SegDataset
from experiment.experiment_2.ori_model_train.PKU.model import BertSeg
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from tqdm import tqdm
from experiment.experiment_2.ori_model_train.PKU.metrics import f1_score, bad_case, output_write, output2res
import torch.nn as nn

def train_epoch(train_loader, model, optimizer, scheduler, epoch, device):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    clip_grad = 5
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels, _ = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_token_starts = batch_token_starts.to(device)
        batch_labels = batch_labels.to(device)
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    print("Epoch: {}, train loss: {}".format(epoch, train_loss))

epoch_num = 5
def load_dataset(path, adv_path):
    train_dataset_path = os.path.join(path, 'pku_training.npz')
    train_data = np.load(train_dataset_path, allow_pickle=True)
    x_train = train_data["words"]
    y_train = train_data["labels"]

    adv_dataset_path = os.path.join(adv_path, 'final_mutants.npz')
    adv_data = np.load(adv_dataset_path, allow_pickle=True)
    x_adv = adv_data["text"]
    y_adv = adv_data["label"]

    x_train = np.concatenate((x_train, x_adv), axis=0)
    y_train = np.concatenate((y_train, y_adv), axis=0)


    test_dataset_path = os.path.join(path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    y_test = test_data["labels"]

    return x_train, x_test, y_train, y_test


def evaluate(dev_loader, model, device, mode='dev'):
    # set model to evaluation mode
    model.eval()
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    id2label = {_id: _label for _label, _id in list(label2id.items())}
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags, ori_data = batch_samples
            # shift tensors to GPU if available
            batch_data = batch_data.to(device)
            batch_token_starts = batch_token_starts.to(device)
            batch_tags = batch_tags.to(device)
            sent_data.extend(ori_data)
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]

            label_masks = batch_tags.gt(-1).to('cpu').numpy()  # get padding mask
            batch_output = batch_output.detach().cpu().numpy()
            batch_tags = batch_tags.to('cpu').numpy()
            for i, indices in enumerate(np.argmax(batch_output, axis=2)):
                pred_tag = []
                for j, idx in enumerate(indices):
                    if label_masks[i][j]:
                        pred_tag.append(id2label.get(idx))
                pred_tags.append(pred_tag)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1, p, r = f1_score(true_tags, pred_tags)
    metrics['f1'] = f1
    metrics['p'] = p
    metrics['r'] = r
    if mode != 'dev':
        bad_case(sent_data, pred_tags, true_tags)
        output_write(sent_data, pred_tags)
        output2res()
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir, device):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # reload weights from restore_dir if specified
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, device)
        val_metrics = evaluate(dev_loader, model, device)
        val_f1 = val_metrics['f1']
        print("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            final_model_save_path = os.path.join(model_dir, 'best_PKU_bert_wwm_chinese_.pt')
            torch.save(model, final_model_save_path)
            # model.module.save_pretrained(model_dir)
            print("--------Save best model!--------")
            patience = 0.0002
            if improve_f1 < patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        patience_num = 4
        min_epoch_num = 5
        if (patience_counter >= patience_num and epoch > min_epoch_num) or epoch == epoch_num:
            print("Best val f1: {}".format(best_val_f1))
            break
    print("Training Finished!")


def train_model(dataset_path, save_model_path, load_model_path, adv_path):
    word_train, word_dev, label_train, label_dev = load_dataset(dataset_path, adv_path)

    train_dataset = SegDataset(word_train, label_train, load_model_path)
    dev_dataset = SegDataset(word_dev, label_dev, load_model_path)

    train_size = len(train_dataset)
    batch_size = 6
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)


    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    model = BertSeg.from_pretrained(load_model_path, num_labels=len(label2id))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    full_fine_tuning = True
    weight_decay = 0.01
    learning_rate = 1e-5
    if full_fine_tuning:
        # model.named_parameters(): [bert, classifier]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': learning_rate * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': learning_rate * 5, 'weight_decay': 0.0}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // batch_size

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=epoch_num * train_steps_per_epoch)

    train(train_loader, dev_loader, model, optimizer, scheduler, save_model_path, device)

if __name__ == "__main__":
    pass
