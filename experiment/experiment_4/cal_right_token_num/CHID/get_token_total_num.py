import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from experiment.experiment_4.cal_right_token_num.CHID.create_dataset import ClozeDataset

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
    d1 = dataset
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

    return dataset, d1

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
            outputs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=1)

            pred_labels.extend(pred_label.tolist())
            true_labels.extend(label.cpu().numpy().tolist())
    results = []
    # print(pred_labels)
    # print(true_labels)
    for num in range(len(pred_labels)):
        if pred_labels[num] == true_labels[num]:
            results.append(True)
        else:
            results.append(False)
    return results


def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
    config = Config()

    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)
    config.BERT_PATH = os.path.join(base_model_path, dataset_name, model_name)

    test_df, test_candidates, test_candidate_ids = load_adv_data(dataset_name, base_adv_path)
    test_dataset, dataset = convert_to_features(test_df, test_candidates, test_candidate_ids, config.pretrained_model)
    get_token(dataset)

def get_token(test_dataset):
    token_list = []
    for num in range(len(test_dataset)):
        temp_list = test_dataset[num]['input_ids'].clone().numpy().tolist()
        token_list.extend(temp_list)
    token_arr = np.unique(np.array(token_list))
    print(len(token_arr))

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    data = pd.read_csv(os.path.join(data_path, 'final_mutants.csv'))
    candidates = np.load(os.path.join(data_path, 'final_mutant_candidate.npy'), allow_pickle=True)
    candidate_ids = np.load(os.path.join(data_path, 'final_mutant_candidate_ids.npy'), allow_pickle=True)
    return data, candidates, candidate_ids

if __name__ == '__main__':

    pass