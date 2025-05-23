import os
import json

import pandas as pd


def save_csv(dataset_path, mode_name):
    file_name = mode_name + '.json'
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, "r", encoding="utf8") as f:
        input_data = json.load(f)["data"]
    context_list = []
    question_list = []
    answer_list = []
    answer_start_list = []
    answer_end_list = []
    all_num = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                answer = qa["answers"][0]['text']
                answer_start = qa["answers"][0]['answer_start']
                answer_end = answer_start + len(answer)
                all_num += 1
                if answer_end > (450 - len(question)):
                    continue
                context = context[:450 - len(question)]
                context_list.append(context)
                question_list.append(question)
                answer_list.append(answer)
                answer_start_list.append(answer_start)
                answer_end_list.append(answer_end)
    merge_dt_dict = {'context': context_list, 'question': question_list, 'answer': answer_list,
                     'answer_start': answer_start_list, 'answer_end': answer_end_list}
    data_df = pd.DataFrame(merge_dt_dict)
    return data_df

def save(dataset_path):
    # train_df = save_csv(dataset_path, 'train')

    test_df = save_csv(dataset_path, 'dev')

    # add_df = save_csv(dataset_path, 'test')

    # train_df = pd.concat([train_df, add_df], axis=0)

    # train_df.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)

    test_df.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)

if __name__ == "__main__":
    pass

