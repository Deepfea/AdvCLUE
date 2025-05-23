import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment.experiment_2.ori_model_train.PKU.create_PKU_dataset import SegDataset
from tqdm import tqdm

def evaluate_data(dev_loader, model_path):
    model = torch.load(model_path)
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
            batch_masks = batch_data.gt(0)
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

def acc_score(y_true, y_pred):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)
    word_acc = float(nb_correct) / float(nb_true)
    return word_acc

def get_entities(seq):
    # print(seq)
    prev_tag = 'O'
    seq += ['O']
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq):
        tag = chunk[0]
        # print(tag)
        if end_of_chunk(prev_tag, tag):
            chunks.append((begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    # print(chunks)
    return chunks


def end_of_chunk(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'E':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'M' and tag == 'B':
        chunk_end = True
    if prev_tag == 'M' and tag == 'S':
        chunk_end = True
    if prev_tag == 'M' and tag == 'O':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'O' and tag == 'M':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'M':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'M':
        chunk_start = True
    if prev_tag == 'E' and tag == 'E':
        chunk_start = True

    return chunk_start

def cal_ASR(model_name, dataset_name, base_adv_path, base_model_path, base_tokenizer_path):

    acc_list = []

    pre_model_path = os.path.join(base_tokenizer_path, model_name)

    adv_data = load_adv_data(dataset_name, base_adv_path)
    x_adv = adv_data["text"]
    y_adv = adv_data["label"]

    train_dataset = SegDataset(x_adv, y_adv, pre_model_path)
    train_loader = DataLoader(train_dataset, batch_size=6, collate_fn=train_dataset.collate_fn)

    load_model_path = os.path.join(base_model_path, dataset_name, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')

    sent_data, pred_tags, true_tags, train_output = evaluate_data(train_loader, load_model_path)

    for num in range(len(pred_tags)):
        temp_acc = acc_score(pred_tags[num], true_tags[num])
        acc_list.append(temp_acc)

    true_num = 0
    for num in range(len(acc_list)):
        if acc_list[num] > 0.85:
            true_num += 1
    result = true_num / float(len(acc_list))
    print(1-result)

def load_adv_data(dataset_name, base_adv_path):
    data_path = os.path.join(base_adv_path, dataset_name, 'final_adv')
    train_data = np.load(os.path.join(data_path, 'final_mutants.npz'), allow_pickle=True)
    return train_data

if __name__ == '__main__':
    pass