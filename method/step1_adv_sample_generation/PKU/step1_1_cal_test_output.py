import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.PKU.create_PKU_dataset import SegDataset

def load_dataset(path):
    test_dataset_path = os.path.join(path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    y_test = test_data["labels"]

    return x_test, y_test


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
                test_output.append(te_output)
                pred_tags.append(pred_tag)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)

    assert len(sent_data) == len(true_tags)

    return sent_data, pred_tags, true_tags, test_output

def evaluate_testing_data(dataset_path, base_model_path, save_path):

    test_dataset_path = os.path.join(dataset_path, 'pku_testing.npz')
    test_data = np.load(test_dataset_path, allow_pickle=True)
    x_test = test_data["words"]
    y_test = test_data["labels"]
    testing_dataset = SegDataset(x_test, y_test)
    testing_loader = DataLoader(testing_dataset, batch_size=6, collate_fn=testing_dataset.collate_fn)
    sent_data, pred_tags, true_tags, test_output = evaluate_data(testing_loader, base_model_path)
    pred_tags = np.array(pred_tags)
    np.save(os.path.join(save_path, 'pre_tags.npy'), pred_tags)
    test_output = np.array(test_output)
    np.save(os.path.join(save_path, 'pre_output.npy'), test_output)
    for i in range(len(pred_tags)):
        print(len(pred_tags[i]))
        print(len(test_output[i]))
    np.save(os.path.join(save_path, 'pku_testing.npz'), test_data)

if __name__ == '__main__':
    pass
