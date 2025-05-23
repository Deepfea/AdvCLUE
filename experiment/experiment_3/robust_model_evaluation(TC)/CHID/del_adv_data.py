import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.CHID.create_dataset import ClozeDataset
from experiment.experiment_2.ori_model_train.CHID.bert_model import BertForClozeBaseline
import transformers
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

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

    flag_list = []
    for num in range(len(pred_labels)):
        if pred_labels[num] == true_labels[num]:
            flag_list.append(True)
        else:
            flag_list.append(False)
    return flag_list

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):
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
    flag_list = get_acc(test_dataloader, model)
    print(flag_list)

    del_num = int(len(flag_list) * 0.5)

    belong_list = []
    tag_list = []
    ori_text_list = []
    str_list = []
    word_list = []
    text_list = []
    candidate_list = []
    groundTruth_list = []
    label_list = []
    type_list = []
    mutant_candidates_list = []
    mutant_candidate_ids_list = []
    error_num_list = []
    temp_num = 0
    for num in range(len(flag_list)):
        if flag_list[num] == 0 and temp_num < del_num:
            temp_num += 1
            continue

        belong_list.append(test_df.loc[num, 'belong'])
        tag_list.append(test_df.loc[num, 'tag'])
        ori_text_list.append(test_df.loc[num, 'ori_text'])
        str_list.append(test_df.loc[num, 'str'])
        word_list.append(test_df.loc[num, 'word'])
        text_list.append(test_df.loc[num, 'text'])
        candidate_list.append(test_df.loc[num, 'candidate'])
        groundTruth_list.append(test_df.loc[num, 'groundTruth'])
        label_list.append(test_df.loc[num, 'label'])
        type_list.append(test_df.loc[num, 'type'])
        mutant_candidates_list.append(test_candidates[num])
        mutant_candidate_ids_list.append(test_candidate_ids[num])
        error_num_list.append(test_df.loc[num, 'error_num'])
    merge_dt_dict = {'belong': belong_list, 'tag': tag_list, 'ori_text': ori_text_list,
                     'str': str_list, 'word': word_list,
                     'text': text_list, 'candidate': candidate_list, 'groundTruth': groundTruth_list,
                     'label': label_list, 'type': type_list, 'error_num': error_num_list
                     }
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(base_adv_path, dataset_name, 'filtered_mutants.csv'), index=False)

    mutant_candidates_arr = np.array(mutant_candidates_list)
    mutant_candidate_ids_arr = np.array(mutant_candidate_ids_list)

    np.save(os.path.join(base_adv_path, dataset_name, 'filtered_mutant_candidates.npy'), mutant_candidates_arr)
    np.save(os.path.join(base_adv_path, dataset_name, 'filtered_mutant_candidate_ids.npy'), mutant_candidate_ids_arr)


def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name)
    data = pd.read_csv(os.path.join(data_path, 'selected_mutants.csv'))
    candidates = np.load(os.path.join(data_path, 'selected_mutant_candidates.npy'), allow_pickle=True)
    candidate_ids = np.load(os.path.join(data_path, 'selected_mutant_candidate_ids.npy'), allow_pickle=True)
    return data, candidates, candidate_ids

if __name__ == '__main__':
    pass
