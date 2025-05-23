import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.CHID.create_dataset import ClozeDataset
from experiment.experiment_2.ori_model_train.CHID.bert_model import BertForClozeBaseline
import transformers
import torch
from transformers import BertTokenizer
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = ''
    TOKENIZER = ''
    pretrained_model = ''


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def convert_to_features(df_data, candidates, candidate_ids, path):
    dataset = ClozeDataset(df_data, candidates, candidate_ids, path)
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

def get_output(test_dataloader, model):
    config = Config()
    model.eval()
    output_list = []
    label_list = []

    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))
        for bi, batch in enumerate(tk0):
            logits = run_one_step(batch, model, config.device)
            label = batch["label"].to(config.device)
            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            outputs = list(outputs)
            output_list.extend(outputs)
            label_list.extend(label.cpu().numpy().tolist())
    output_arr = np.array(output_list)
    return output_arr, label_list


def get_pic1(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):

    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    config.TOKENIZER = BertTokenizer.from_pretrained(config.pretrained_model, lowercase=True)

    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    print(len(adv_data))
    adv_candidates = np.load(os.path.join(load_adv_path, 'final_mutant_candidate.npy'), allow_pickle=True)
    adv_candidate_ids = np.load(os.path.join(load_adv_path, 'final_mutant_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(adv_data, adv_candidates, adv_candidate_ids, config.pretrained_model)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    trained_model_path = os.path.join(base_model_path, 'ori_model', dataset_name, model_name, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    output_arr1, label_list1 = get_output(test_dataloader, model)

    trained_model_path = os.path.join(base_model_path, 'retrained_model', dataset_name, model_name, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    output_arr2, label_list2 = get_output(test_dataloader, model)

    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):

    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    config.TOKENIZER = BertTokenizer.from_pretrained(config.pretrained_model, lowercase=True)

    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    load_adv_path = base_adv_path
    adv_data = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    print(len(adv_data))
    adv_candidates = np.load(os.path.join(load_adv_path, 'adv_candidates.npy'), allow_pickle=True)
    adv_candidate_ids = np.load(os.path.join(load_adv_path, 'adv_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(adv_data, adv_candidates, adv_candidate_ids, config.pretrained_model)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    trained_model_path = os.path.join(base_model_path, 'ori_model', dataset_name, model_name, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)
    output_arr1, label_list1 = get_output(test_dataloader, model)

    output_arr2, label_list2 = get_output(test_dataloader, model)


    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

if __name__ == "__main__":
    pass


