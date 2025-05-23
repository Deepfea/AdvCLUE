import numpy as np
import pandas as pd
from method.step1_adv_sample_generation.LCQMC.load_token import load_token
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.CHID.create_dataset import ClozeDataset
from method.step1_adv_sample_generation.CHID.bert_model import BertForClozeBaseline
import transformers
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer

idiom_vocab = eval(open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 8
    BERT_PATH = '/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese'
    # BERT_PATH = project_dir + "/pretrained_models/ernie_based_pretrained"
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

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

def cal_output(dataset_path, base_model_path, save_path):
    config = Config()

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    print(test_df)
    test_candidates = np.load(os.path.join(dataset_path, 'test_candidates.npy'), allow_pickle=True)
    test_candidate_ids = np.load(os.path.join(dataset_path, 'test_candidate_ids.npy'), allow_pickle=True)
    test_dataset = convert_to_features(test_df, test_candidates, test_candidate_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model_config = transformers.BertConfig.from_pretrained(config.pretrained_model)
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=config.pretrained_model,
                                                 config=model_config, idiom_num=len(idiom_vocab))

    trained_model_path = os.path.join(base_model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))
    model.to(config.device)

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
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'testing_data_output.npy'), output_arr)

    print(output_arr)
    print(output_arr.shape)
    test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    np.save(os.path.join(save_path, 'test_candidates.npy'), test_candidates)
    np.save(os.path.join(save_path, 'test_candidate_ids.npy'), test_candidate_ids)
    # return output_arr

    pre_labels = []
    for num in range(len(output_arr)):
        temp_arr = output_arr[num]

        pre_labels.append(np.argmax(temp_arr))
    true_labels = test_df['label']
    all_num = len(true_labels)
    true_num = 0
    for num in range(len(pre_labels)):
        if pre_labels[num] == true_labels[num]:
            true_num += 1
    print(float(true_num) / float(all_num))



if __name__ == '__main__':
    pass
