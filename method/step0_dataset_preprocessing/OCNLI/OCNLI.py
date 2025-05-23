import os
import pandas as pd


def read_ocnli(data_dir, is_train):
    labels_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 3}

    file_name = os.path.join(data_dir, 'train.50k.json' if is_train else 'dev.json')
    rows = pd.read_json(file_name, lines=True)

    premises = [sentence1 for sentence1 in rows['sentence1']]  # 前提
    hypotheses = [sentence2 for sentence2 in rows['sentence2']]  # 假设
    labels = [label for label in rows['label']]

    premises_list = []
    hypotheses_list = []
    label_list = []
    for num in range(len(labels)):
        if labels[num] != '-':
            premises_list.append(premises[num])
            hypotheses_list.append(hypotheses[num])
            label_list.append(int(labels_map[labels[num]]))
    print(label_list)
    print(len(premises_list))
    print(len(hypotheses_list))
    print(len(label_list))

    merge_dt_dict = {'premise': premises_list, 'hypothese': hypotheses_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    if is_train:
        data_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    else:
        data_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    return premises, hypotheses, labels





if __name__ == "__main__":
    pass

