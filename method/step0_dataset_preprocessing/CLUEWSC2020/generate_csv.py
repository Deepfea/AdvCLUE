import os

import pandas as pd
import json

def get_wsc_json(file):
    with open(file, "r") as f:
        datas = []
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas

def transfers_wsc_df(df):
    df["span1_begin"] = df.apply(
        lambda row: _get_begin_range(row["target"]["span1_index"]), axis=1
    )
    df["span1_end"] = df.apply(
        lambda row: _get_begin_range(
            row["target"]["span1_index"], row["target"]["span1_text"]
        ),
        axis=1,
    )
    df["span2_begin"] = df.apply(
        lambda row: _get_begin_range(row["target"]["span2_index"]), axis=1
    )
    df["span2_end"] = df.apply(
        lambda row: _get_begin_range(
            row["target"]["span2_index"], row["target"]["span2_text"]
        ),
        axis=1,
    )
    df["label_id"] = df["label"].apply(lambda x: 1 if x == "true" else 0)
    return df


def _get_begin_range(index, length=""):
    return index + len(length)

def _get_end_range(index, length=""):
    return index + len(length) - 1

def save_csv(dataset_path):
    datas = get_wsc_json("/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020/train.json")
    train_df = pd.DataFrame(datas)
    train_df = transfers_wsc_df(train_df)

    train_df.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)

    print(train_df)

    datas = get_wsc_json("/media/usr/external/home/usr/project/project3_data/dataset/CLUEWSC2020/dev.json")
    dev_df = pd.DataFrame(datas)
    dev_df = transfers_wsc_df(dev_df)
    dev_df.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)

    print(train_df)



if __name__ == "__main__":
    pass

