import numpy as np
import pandas as pd
from method.step2_calculate_metric.CMRC2018.load_token import get_token
import os
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from experiment.experiment_2.ori_model_train.CMRC2018.create_dataset import SquadDataset
import torch
from experiment.experiment_3.TSE.cal_T_SNE_value import cal_T_SNE

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = ''
    pretrained_model = ''

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(model, dev_loader):
    model.eval()
    start_list = []
    end_list = []
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
        start_list.extend(outputs.start_logits.cpu().clone().detach().numpy().tolist())
        # end_list.extend(outputs.end_logits.cpu().clone().detach().numpy().tolist())
    start_list = np.array(start_list)
    # end_list = np.array(end_list)

    output_arr = []
    label_arr = []
    for num in range(len(start_list)):
        temp_output = start_list[num]
        x = softmax(temp_output)
        output_arr.append(x)
        label_arr.append(0)
    output_arr = np.array(output_arr)
    label_arr = np.array(label_arr)
    return output_arr, label_arr

def get_pic(model_name, dataset_name, base_tokenizer_path, base_adv_path, base_model_path, save_path):
    config = Config()
    config.pretrained_model = os.path.join(base_tokenizer_path, model_name)

    load_adv_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    test_df = pd.read_csv(os.path.join(load_adv_path, 'final_mutants.csv'))
    test_encodings = get_token(test_df, config.pretrained_model)
    test_dataset = SquadDataset(test_encodings, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    load_model_path = os.path.join(base_model_path, 'ori_model', dataset_name, model_name)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    output_arr1, label_list1 = evaluate(model, test_loader)

    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    load_model_path = os.path.join(base_model_path, 'retrained_model', dataset_name, model_name)
    checkpoint = torch.load(os.path.join(load_model_path, 'best_' + dataset_name + '_' + model_name + '.model'))
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    output_arr2, label_list2 = evaluate(model, test_loader)

    cal_T_SNE(dataset_name, model_name, output_arr1, label_list1, output_arr2, label_list2, save_path)

if __name__ == "__main__":
    pass
