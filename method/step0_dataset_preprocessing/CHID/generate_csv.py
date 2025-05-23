import json
import os
import re

import numpy as np
import pandas as pd

idiom_vocab = eval(
            open('/media/usr/external/home/usr/project/project3_data/dataset/CHID/idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
# print(idiom_vocab)

def save_csv(path, name):
    data_id_list = []
    tag_list = []
    text_list = []
    candidate_list = []
    groundTruth_list = []


    dataset_path = os.path.join(path, name + '.txt')



    data_id = 0
    with open(dataset_path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            cur_data = json.loads(line)
            groundTruth = cur_data["groundTruth"]
            candidates = cur_data["candidates"]
            content = cur_data["content"]
            realCount = cur_data["realCount"]
            for i in range(realCount):
                content = content.replace("#idiom#", f"#idiom{i+1}#", 1)
            tags = re.findall("#idiom\d+#", content)
            # print(content)
            # print(tags)
            # print(groundTruth)

            # if len(content) > 200:
            #     continue
            for tag_num in range(len(tags)):
                tmp_context = content
                for other_tag_num in range(len(tags)):
                    if tag_num != other_tag_num:
                        tmp_context = tmp_context.replace(tags[other_tag_num], groundTruth[other_tag_num])
                data_id_list.append(data_id)
                tag_list.append(tags[tag_num])
                text_list.append(tmp_context)
                candidate_list.append(candidates[tag_num])
                groundTruth_list.append(groundTruth[tag_num])
            data_id += 1
    merge_dt_dict = {'data_id': data_id_list, 'tag': tag_list, 'text': text_list, 'candidate': candidate_list, 'groundTruth':groundTruth_list}
    candidate_arr = np.array(candidate_list)

    np.save(os.path.join(path, name + '_candidates.npy'), candidate_arr)

    data_df = pd.DataFrame(merge_dt_dict)
    print(len(data_df))


    label_list = []
    for num in range(len(data_df)):
        temp_label = list(candidate_arr[num]).index(data_df.loc[num, 'groundTruth'])
        label_list.append(temp_label)
    data_df['label'] = label_list

    data_df.to_csv(os.path.join(path, name + '.csv'), index=False)

    candidate_ids_list = []
    for num in range(len(candidate_arr)):
        temp_ids = [idiom_vocab[each] for each in candidate_arr[num]]
        candidate_ids_list.append(temp_ids)
    candidate_ids_arr = np.array(candidate_ids_list)

    np.save(os.path.join(path, name + '_candidate_ids.npy'), candidate_ids_arr)

    print(data_df)
    print(candidate_arr)
    print(candidate_arr.shape)
    print(candidate_ids_arr)
    print(candidate_ids_arr.shape)


def get_all_dataset_json(dataset_path):
    name = 'train'
    save_csv(dataset_path, name)

    name = 'test'
    save_csv(dataset_path, name)



if __name__ == "__main__":
    pass
