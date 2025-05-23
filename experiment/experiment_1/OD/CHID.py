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

def get_output(test_dataloader, model):
    config = Config()
    model.eval()
    output_list = []
    # Turn of gradient calculations
    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))

        for bi, batch in enumerate(tk0):
            logits = run_one_step(batch, model, config.device)

            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            outputs = list(outputs)
            output_list.extend(outputs)

    output_arr = np.array(output_list)
    return output_arr

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
    output1 = get_output(test_dataloader, model)

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(dataset_path, 'test_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    output2 = get_output(test_dataloader, model)

    output = np.concatenate((output1, output2), axis=0)
    output_arr = np.array(output)
    np.save(os.path.join(save_path, 'output.npy'), output_arr)
    print(len(output_arr))

    result = get_gini(output_arr)
    print(result)

def get_gini(output_list):
    value_list = []
    for output_num in range(len(output_list)):
        value = 0
        temp_output = output_list[output_num]
        for num in range(len(temp_output)):
            value = value + temp_output[num] * temp_output[num]
        value_list.append(value)
    value_arr = np.array(value_list)
    result = np.std(value_arr)
    return result

if __name__ == '__main__':
    pass
