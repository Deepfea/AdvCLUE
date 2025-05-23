import numpy as np
import os
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
    pre_output = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
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
                pre_output.append(te_output)
                pred_tags.append(pred_tag)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    assert len(sent_data) == len(true_tags)

    return sent_data, pred_tags, true_tags, pre_output

def get_data_without_S(data_x, data_y, data_output, i):
    temp_x = []
    temp_y = []
    temp_output = []
    temp_word = data_x[i]
    # print(data_x)
    # print(len(data_x))
    # print(data_y)
    # print(len(data_y))
    # print(data_output)
    # print(len(data_output))
    for num in range(len(data_x)):
        if num != i:
            temp_x.append(data_x[num])
            temp_y.append(data_y[num])
            temp_output.append(data_output[num])
    return temp_x, temp_y, temp_output, temp_word

def get_data_without_B(data_x, data_y, data_output, i):
    temp_x = []
    temp_y = []
    temp_output = []
    temp_word = ''
    if i == len(data_y) - 1:
        j = i
    else:
        j = i + 1
        while data_y[j] == 'M' and j != (len(data_y) - 1):
            j += 1
        if j == (len(data_y) - 1):
            if data_y[j] == 'B' or data_y[j] == 'S':
                j -= 1
        else:
            if data_y[j] != 'E':
                j -= 1
    for num in range(len(data_x)):
        if num >= i and num <= j:
            temp_word += data_x[num]
            continue
        temp_x.append(data_x[num])
        temp_y.append(data_y[num])
        temp_output.append(data_output[num])
    return temp_x, temp_y, temp_output, temp_word, j

def get_data_without_word(data_x, data_y, data_output):
    data_without_word_x = []
    data_without_word_y = []
    data_without_word_output = []
    word = []
    position = []
    i = 0
    while i < len(data_x):
        if data_y[i] == 'S':
            temp_poi = []
            temp_x, temp_y, temp_output, temp_word = get_data_without_S(data_x, data_y, data_output, i)
            data_without_word_x.append(temp_x)
            data_without_word_y.append(temp_y)
            data_without_word_output.append(temp_output)
            word.append(temp_word)
            temp_poi.append(i)
            temp_poi.append(i)
            position.append(temp_poi)
            i += 1
        elif data_y[i] == 'B':
            temp_poi = []
            temp_x, temp_y, temp_output, temp_word, j = get_data_without_B(data_x, data_y, data_output, i)
            data_without_word_x.append(temp_x)
            data_without_word_y.append(temp_y)
            data_without_word_output.append(temp_output)
            word.append(temp_word)
            temp_poi.append(i)
            temp_poi.append(j)
            position.append(temp_poi)
            i = j + 1
        else:
            i += 1
    return data_without_word_x, data_without_word_y, data_without_word_output, word, position

def cal_score(ori_output, pre_output, data_without_word_y):
    scores = []
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    for num in range(len(ori_output)):
        score = 0
        for tag_num in range(len(ori_output[num])):
            temp_label = data_without_word_y[num][tag_num]
            temp_id = label2id.get(temp_label)
            score += ori_output[num][tag_num][temp_id] - pre_output[num][tag_num][temp_id]
        scores.append(score)
    scores = np.array(scores)
    return scores

def cal_word_importance(save_path, dataset_path, base_model_path):
    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]

    pre_tags_path = os.path.join(save_path, 'pre_tags.npy')
    pre_tags = np.load(pre_tags_path, allow_pickle=True)

    pre_output_path = os.path.join(save_path, 'pre_output.npy')
    true_output = np.load(pre_output_path, allow_pickle=True)

    words_list = []
    scores_list = []
    position_list = []
    for data_num in tqdm(range(len(x_test))):
        # print(data_num)
        temp_x = x_test[data_num]
        temp_y = pre_tags[data_num]
        temp_output = true_output[data_num]
        # print(len(temp_output))
        data_without_word_x, data_without_word_y, data_without_word_output, words, positions = get_data_without_word(temp_x, temp_y, temp_output)
        # print(positions)
        testing_dataset = SegDataset(data_without_word_x, data_without_word_y)
        testing_loader = DataLoader(testing_dataset, batch_size=6, collate_fn=testing_dataset.collate_fn)
        sent_data_arr, pred_tags_arr, true_tags_arr, pre_output = evaluate_data(testing_loader, base_model_path)
        scores = cal_score(data_without_word_output, pre_output, data_without_word_y)
        words_list.append(words)
        scores_list.append(scores)
        position_list.append(positions)
        # print(len(words))
        # print(len(scores))
        # break
    words_arr = np.array(words_list)
    scores_arr = np.array(scores_list)
    positions_arr = np.array(position_list)
    np.save(os.path.join(save_path, 'words.npy'), words_arr)
    np.save(os.path.join(save_path, 'scores.npy'), scores_arr)
    np.save(os.path.join(save_path, 'positions.npy'), positions_arr)
    # print(words)
    # print(scores)
    # print(positions_arr)


if __name__ == '__main__':
    pass
