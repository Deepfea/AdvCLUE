import numpy as np
import os
import torch
from method.step1_adv_sample_generation.CLUEWSC2020.dataset import get_dev_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT
from tqdm import tqdm

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def get_output(dev_dataloader, model):
    config = Config()
    model.eval()
    test_flag = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(dev_dataloader, 0)):
            input_ids = data["input_ids"].to(config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(config.device, dtype=torch.long)
            targets = data["labels"].to(config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(config.device, dtype=torch.long)
            input = (input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end,)
            outputs = model(input)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = outputs
                total_label = targets
            else:
                total_npy = np.concatenate((total_npy, outputs), axis=0)
                total_label = np.concatenate((total_label, targets), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(total_npy)
    return output_arr

def evaluate_testing_data(dataset_path, base_model_path, save_path):
    config = Config()
    config.pretrained_model = base_model_path
    model = BERT(config)
    model = model.to(config.device)

    dataset_path_path = os.path.join(dataset_path, 'train.csv')
    dev_dataloader, def_df = get_dev_dataloader(dataset_path_path, config.max_length, config.batch_size)
    output1 = get_output(dev_dataloader, model)

    dataset_path_path = os.path.join(dataset_path, 'test.csv')
    dev_dataloader, def_df = get_dev_dataloader(dataset_path_path, config.max_length, config.batch_size)
    output2 = get_output(dev_dataloader, model)

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

if __name__ == "__main__":
    pass
