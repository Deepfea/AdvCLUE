import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.CHID.create_dataset import ClozeDataset
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
import transformers
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/CHID/bert_base_chinese'
    # BERT_PATH = project_dir + "/pretrained_models/ernie_based_pretrained"
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/experiment/experiment1/advCLUE/CHID/bert_base_chinese"

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def convert_to_features(df_data, candidates, candidate_ids):
    dataset = ClozeDataset(df_data, candidates, candidate_ids)
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

def get_acc(valid_data_loader, model):
    config = Config()
    device = config.device
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

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()
    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    trained_model_path = os.path.join(base_model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

    test_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_candidates = np.load(os.path.join(dataset_path, 'train_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(dataset_path, 'train_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    acc1 = get_acc(test_dataloader, model)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(dataset_path, 'test_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    acc2 = get_acc(test_dataloader, model)

    print((acc1 + acc2) / 2.0)

if __name__ == '__main__':
    pass
