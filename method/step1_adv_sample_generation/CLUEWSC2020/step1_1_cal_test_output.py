import numpy as np
import os
import torch
from dataset import get_wsc_dataloader, get_dev_dataloader
from method.step1_adv_sample_generation.CLUEWSC2020.model import BERT


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = 25
    max_length = 200
    pretrained_model = ''

def evaluate_testing_data(dataset_path, base_model_path, save_path):
    config = Config()
    config.pretrained_model = base_model_path
    dataset_path_path = os.path.join(dataset_path, 'test.csv')
    dev_dataloader, def_df = get_dev_dataloader(dataset_path_path, config.max_length, config.batch_size)
    model = BERT(config)
    model = model.to(config.device)
    model.eval()
    test_flag = 0
    with torch.no_grad():
        for _, data in enumerate(dev_dataloader, 0):
            print(_)
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
    print(total_label)
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(total_npy)
    # print(output_arr.shape)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'testing_data_output.npy'), output_arr)
    count = 0
    for num in range(len(output_arr)):
        print(np.argmax(output_arr[num]))
        print(def_df.loc[num, 'label_id'])
        if np.argmax(output_arr[num]) != def_df.loc[num, 'label_id']:
            count += 1
    print(count)
    print(len(def_df))
    def_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)

if __name__ == "__main__":
    pass
