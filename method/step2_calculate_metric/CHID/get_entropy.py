import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from method.step2_calculate_metric.CHID.create_dataset import ClozeDataset, ori_ClozeDataset
from experiment.experiment_2.ori_model_train.CHID.bert_model import BertForClozeBaseline
import transformers


idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = ''
    pretrained_model = ''

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def convert_to_features(df_data, candidates, candidate_ids, BERT_PATH):
    dataset = ClozeDataset(df_data, candidates, candidate_ids, BERT_PATH)
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

def convert_to_ori_features(df_data, candidates, candidate_ids, BERT_PATH):
    dataset = ori_ClozeDataset(df_data, candidates, candidate_ids, BERT_PATH)
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
    outputs = []
    with torch.no_grad():
        tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))
        for bi, batch in enumerate(tk0):
            logits = run_one_step(batch, model, device)
            # print(logits)
            output = logits.cpu().detach().numpy()
            outputs.extend(output)
    return outputs

def get_entropy(outputs1, outputs2):
    results = []
    for num in range(len(outputs1)):
        temp_output1 = outputs1[num]
        temp_output2 = outputs2[num]
        dot_product = np.dot(temp_output1, temp_output2)
        norm_temp_output1 = np.linalg.norm(temp_output1)
        norm_temp_output2 = np.linalg.norm(temp_output2)
        similarity = dot_product / (norm_temp_output1 * norm_temp_output2)
        results.append(similarity)
    results = np.array(results)
    total_sum = np.sum(results)
    final_result = 0
    for result_num in range(len(results)):
        temp_result = - results[result_num] / total_sum * np.log2(results[result_num] / total_sum)
        if np.isnan(temp_result):
            temp_result = 1
        final_result += temp_result
    return final_result

def cal_entropy(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()

    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    config.BERT_PATH = os.path.join(base_model_path, dataset_name, model_name)

    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    trained_model_path = os.path.join(config.BERT_PATH, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

    test_df, test_candidates, test_candidate_ids = load_adv_data(dataset_name, base_adv_path)
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids, config.pretrained_model)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    outputs1 = get_acc(test_dataloader, model)

    test_df, test_candidates, test_candidate_ids = load_adv_data(dataset_name, base_adv_path)
    test_dataset = convert_to_ori_features(test_df, test_candidates, test_candidate_ids, config.pretrained_model)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    outputs2 = get_acc(test_dataloader, model)

    result = get_entropy(outputs1, outputs2)
    print(result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    candidates = np.load(os.path.join(data_path, 'final_mutant_candidate.npy'), allow_pickle=True)
    candidate_ids = np.load(os.path.join(data_path, 'final_mutant_candidate_ids.npy'), allow_pickle=True)
    return data, candidates, candidate_ids

if __name__ == '__main__':
    pass