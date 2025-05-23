import json
import os
import random
import pandas as pd


def get_data(dataset_path):
    dataset = [[], []]

    # 文章，问题，4个选项（不足4个，补充‘’），答案

    for sid in range(2):
        data = []
        for subtask in ["d", "m"]:
            with open(dataset_path + "/c3-" + subtask + "-" + ["train.json", "dev.json"][sid], "r",
                      encoding="utf8") as f:
                data += json.load(f)
        if sid == 0:
            random.shuffle(data)
        for i in range(len(data)):
            for j in range(len(data[i][1])):
                d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                for k in range(len(data[i][1][j]["choice"])):
                    d += [data[i][1][j]["choice"][k].lower()]
                for k in range(len(data[i][1][j]["choice"]), 4):
                    d += ['']
                d += [data[i][1][j]["answer"].lower()]
                dataset[sid] += [d]
    train_df, test_df = get_df(dataset)
    train_df.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)


def get_df(dataset):
    text_list = []
    question_list = []
    candidate_1_list = []
    candidate_2_list = []
    candidate_3_list = []
    candidate_4_list = []
    answer_list = []
    temp_data = dataset[0]
    for num in range(len(temp_data)):
        text_list.append(temp_data[num][0])
        question_list.append(temp_data[num][1])
        candidate_1_list.append(temp_data[num][2])
        candidate_2_list.append(temp_data[num][3])

        if temp_data[num][4] != '':
            candidate_3_list.append(temp_data[num][4])
        else:
            candidate_3_list.append('none')

        if temp_data[num][5] != '':
            candidate_4_list.append(temp_data[num][5])
        else:
            candidate_4_list.append('none')

        answer_list.append(temp_data[num][6])
    merge_dt_dict = {'text': text_list, 'question': question_list, 'candidate_0': candidate_1_list, 'candidate_1': candidate_2_list,
                     'candidate_2': candidate_3_list, 'candidate_3': candidate_4_list, 'answer': answer_list}
    train_df = pd.DataFrame(merge_dt_dict)

    text_list = []
    question_list = []
    candidate_1_list = []
    candidate_2_list = []
    candidate_3_list = []
    candidate_4_list = []
    answer_list = []
    temp_data = dataset[1]
    for num in range(len(temp_data)):
        text_list.append(temp_data[num][0])
        question_list.append(temp_data[num][1])
        candidate_1_list.append(temp_data[num][2])
        candidate_2_list.append(temp_data[num][3])
        if temp_data[num][4] != '':
            candidate_3_list.append(temp_data[num][4])
        else:
            candidate_3_list.append('none')

        if temp_data[num][5] != '':
            candidate_4_list.append(temp_data[num][5])
        else:
            candidate_4_list.append('none')
        answer_list.append(temp_data[num][6])
    merge_dt_dict = {'text': text_list, 'question': question_list, 'candidate_0': candidate_1_list,
                     'candidate_1': candidate_2_list,
                     'candidate_2': candidate_3_list, 'candidate_3': candidate_4_list, 'answer': answer_list}
    test_df = pd.DataFrame(merge_dt_dict)
    return train_df, test_df

if __name__ == "__main__":
    pass
