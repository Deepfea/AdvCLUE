import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.PKU.create_PKU_dataset import SegDataset
from tqdm import tqdm

def evaluate_data(dev_loader, model_path):
    model = torch.load(os.path.join(model_path, 'best_PKU_bert_base_chinese_.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    id2label = {_id: _label for _label, _id in list(label2id.items())}
    true_tags = []
    pred_tags = []
    sent_data = []
    test_output = []
    with torch.no_grad():
        for idx, batch_samples in tqdm(enumerate(dev_loader)):
            batch_data, batch_token_starts, batch_tags, ori_data = batch_samples
            batch_data = batch_data.to(device)
            batch_token_starts = batch_token_starts.to(device)
            batch_tags = batch_tags.to(device)
            sent_data.extend(ori_data)
            batch_masks = batch_data.gt(0)  # get padding mask
            # shape: (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]

            label_masks = batch_tags.gt(-1).to('cpu').numpy()  # get padding mask
            batch_output = batch_output.detach().cpu().numpy()
            batch_tags = batch_tags.to('cpu').numpy()
            for i, indices in enumerate(np.argmax(batch_output, axis=2)):
                pred_tag = []
                te_output = []
                for j, idx in enumerate(indices):
                    if label_masks[i][j]:
                        pred_tag.append(id2label.get(idx))
                        te_output.append(batch_output[i][j])
                test_output.append(te_output)
                pred_tags.append(pred_tag)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    assert len(sent_data) == len(true_tags)

    return sent_data, pred_tags, true_tags, test_output

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def evaluate_testing_data(dataset_path, base_model_path, save_path):

    train_dataset_path = os.path.join(dataset_path, 'pku_training.npz')
    train_data = np.load(train_dataset_path, allow_pickle=True)
    x_train = train_data["words"]
    y_train = train_data["labels"]
    train_dataset = SegDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=6, collate_fn=train_dataset.collate_fn)
    sent_data, pred_tags, true_tags, train_output = evaluate_data(train_loader, base_model_path)

    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    y_test = test_data["labels"]
    testing_dataset = SegDataset(x_test, y_test)
    testing_loader = DataLoader(testing_dataset, batch_size=6, collate_fn=testing_dataset.collate_fn)
    sent_data, pred_tags, true_tags, test_output = evaluate_data(testing_loader, base_model_path)

    total_output_list = []
    total_output_list.extend(train_output)
    total_output_list.extend(test_output)

    output_list = []
    for num in range(len(total_output_list)):
        output_list.append([])
        temp_output = total_output_list[num]
        for temp_num in range(len(temp_output)):
            temp_npy = temp_output[temp_num]
            temp_npy = softmax(temp_npy)
            output_list[num].append(temp_npy)
    output_arr = np.array(output_list)
    np.save(os.path.join(save_path, 'output.npy'), output_arr)

    print(output_arr)
    result = get_gini(output_arr)
    print(result)

def get_gini(output_list):
    value_list = []
    for output_num in range(len(output_list)):
        temp_output = output_list[output_num]
        temp_value_list = []
        for word_num in range(len(temp_output)):
            value = 0
            temp_word_output = temp_output[word_num]
            for num in range(len(temp_word_output)):
                value = value + temp_word_output[num] * temp_word_output[num]
            temp_value_list.append(value)
        temp_value_list = np.array(temp_value_list)
        final_value = np.average(temp_value_list)
        value_list.append(final_value)
    value_arr = np.array(value_list)
    print(value_arr)
    result = np.std(value_arr)
    return result

if __name__ == '__main__':
    pass
