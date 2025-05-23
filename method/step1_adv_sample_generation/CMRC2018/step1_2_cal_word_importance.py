import numpy as np
import os
import pandas as pd
import torch

from method.step1_adv_sample_generation.CMRC2018.seg_data_to_words import seg_data
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
from method.step1_adv_sample_generation.CMRC2018.create_dataset import SquadDataset
from method.step1_adv_sample_generation.CMRC2018.load_token import get_token

from tqdm import tqdm


def segmentation(data_path):
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr, position_arr = seg_data(test_data)
    np.save(os.path.join(data_path, 'seg_arr.npy'), seg_arr)
    np.save(os.path.join(data_path, 'position_arr.npy'), position_arr)


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def get_del_text(text, poi):
    final_str = ''
    for num in range(len(text)):
        if num >= poi[0] and num < poi[1]:
            continue
        final_str += text[num]
    return final_str

def find_start_and_end(ori_start, ori_end, word, poi):
    if poi[1] <= ori_start:
        final_start = ori_start - len(word)
        final_end = ori_end - len(word)
    else:
        final_start = ori_start
        final_end = ori_end
    return final_start, final_end

def gen_new_sentence(data_path):
    belong_list = []
    belong_text = []
    str_list = []
    fact_a_list = []
    fact_b_list = []
    answer_start_list = []
    answer_end_list = []
    position_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·', '&']

    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    seg_arr = np.load(os.path.join(data_path, 'seg_arr.npy'), allow_pickle=True)
    position_arr = np.load(os.path.join(data_path, 'position_arr.npy'), allow_pickle=True)
    # print(position_arr)

    for data_num in range(len(seg_arr)):
        temp_segs = seg_arr[data_num]
        temp_pos = position_arr[data_num]

        ori_fact_a = test_data.loc[data_num, 'context']
        ori_fact_b = test_data.loc[data_num, 'question']
        answer_start = test_data.loc[data_num, 'answer_start']
        answer_end = test_data.loc[data_num, 'answer_end']
        segs_a = temp_segs[0]
        poi_a = temp_pos[0]
        segs_b = temp_segs[1]
        poi_b = temp_pos[1]


        for seg_num in range(len(segs_a)):
            temp_seg = segs_a[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            # temp_fact = ori_fact_a.replace(temp_seg, '')
            temp_fact = get_del_text(ori_fact_a, poi_a[seg_num])
            position_list.append(poi_a[seg_num])
            belong_list.append(data_num)
            belong_text.append('a')
            str_list.append(temp_seg)
            fact_a_list.append(temp_fact)
            fact_b_list.append(ori_fact_b)
            temp_start, temp_end = find_start_and_end(answer_start, answer_end, temp_seg, poi_a[seg_num])
            answer_start_list.append(temp_start)
            answer_end_list.append(temp_end)

        for seg_num in range(len(segs_b)):
            temp_seg = segs_b[seg_num]
            if temp_seg in remove_flag:
                continue
            if temp_seg == ' ':
                continue
            temp_fact = get_del_text(ori_fact_b, poi_b[seg_num])
            # temp_fact = ori_fact_b.replace(temp_seg, '')
            position_list.append(poi_b[seg_num])
            belong_list.append(data_num)
            belong_text.append('b')
            str_list.append(temp_seg)
            fact_a_list.append(ori_fact_a)
            fact_b_list.append(temp_fact)
            temp_start = answer_start
            temp_end = answer_end
            answer_start_list.append(temp_start)
            answer_end_list.append(temp_end)

    merge_dt_dict = {'belong': belong_list, 'belong_text': belong_text, 'str': str_list, 'position': position_list,'context': fact_a_list,
                     'question': fact_b_list, 'answer_start': answer_start_list, 'answer_end': answer_end_list}
    data_df = pd.DataFrame(merge_dt_dict)
    print(data_df)
    return data_df

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    max_length = 510
    batch_size = 16
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def get_output(model, dev_loader):
    acc_start_sum = 0.0
    acc_end_sum = 0.0
    model.eval()
    start_list = []
    end_list = []
    for idx, batch in enumerate(tqdm(dev_loader)):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        temp_start = outputs.start_logits.detach().cpu().clone().numpy()
        start_list.extend(temp_start)
        temp_end = outputs.end_logits.detach().cpu().clone().numpy()
        end_list.extend(temp_end)

        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)
        # print((start_pred == start_positions))
        # print((start_pred == start_positions).float().mean())
        acc_start = (start_pred == start_positions).float().mean()
        acc_end = (end_pred == end_positions).float().mean()

        acc_start_sum += acc_start
        acc_end_sum += acc_end
    # print(acc_start_sum / len(dev_loader))
    # print(acc_end_sum / len(dev_loader))

    return start_list, end_list

def get_start_and_end_confidence(df, start_output, end_output):
    encodings = get_token(df)
    starts = encodings['start_positions'].numpy()
    ends = encodings['end_positions'].numpy()
    start_confidence_list = []
    end_confidence_list = []
    for num in range(len(df)):
        start_index = starts[num]
        end_index = ends[num]
        temp_start_confidence = start_output[num][start_index]
        temp_end_confidence = end_output[num][end_index]
        start_confidence_list.append(temp_start_confidence)
        end_confidence_list.append(temp_end_confidence)
    start_confidence_arr = np.array(start_confidence_list)
    end_confidence_arr = np.array(end_confidence_list)

    # print(df.loc[296])
    # start_pre = encodings.token_to_chars(296, encodings['start_positions'][296])
    # end_pre = encodings.token_to_chars(296, encodings['end_positions'][296])
    # doc = df.loc[296, 'context']
    # print(doc[start_pre.start: end_pre.end])

    return start_confidence_arr, end_confidence_arr

def cal_word_importance(save_path, base_model_path):
    config = Config()
    data_path = os.path.join(save_path, 'source_data')
    segmentation(data_path)

    new_data_df = gen_new_sentence(data_path)
    new_data_df.to_csv(os.path.join(data_path, 'temp_data.csv'), index=False)

    new_data_encodings = get_token(new_data_df)
    new_data_dataset = SquadDataset(new_data_encodings, config)
    new_data_loader = DataLoader(new_data_dataset, batch_size=config.batch_size, shuffle=False)
    config = Config()
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    #
    model.eval()
    starts, ends = get_output(model, new_data_loader)

    start_list = []
    for temp_num in range(len(starts)):
        temp_npy = starts[temp_num]
        temp_npy = softmax(temp_npy)
        start_list.append(temp_npy)
    start_arr = np.array(start_list)

    end_list = []
    for temp_num in range(len(ends)):
        temp_npy = ends[temp_num]
        temp_npy = softmax(temp_npy)
        end_list.append(temp_npy)
    end_arr = np.array(end_list)


    np.save(os.path.join(data_path, 'new_data_start_output.npy'), start_arr)
    np.save(os.path.join(data_path, 'new_data_end_output.npy'), end_arr)

    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_data_start_output = np.load(os.path.join(data_path, 'test_data_start_output.npy'))
    test_data_end_output = np.load(os.path.join(data_path, 'test_data_end_output.npy'))
    test_data_start_confidence, test_data_end_confidence = get_start_and_end_confidence(test_data, test_data_start_output, test_data_end_output)

    new_data = pd.read_csv(os.path.join(data_path, 'temp_data.csv'))
    new_data_start_output = np.load(os.path.join(data_path, 'new_data_start_output.npy'))
    new_data_end_output = np.load(os.path.join(data_path, 'new_data_end_output.npy'))
    new_data_start_confidence, new_data_end_confidence = get_start_and_end_confidence(new_data,
                                                                                        new_data_start_output,
                                                                                        new_data_end_output)
    importance_value = []
    for data_num in range(len(new_data)):
        temp_value_1 = test_data_start_confidence[new_data.loc[data_num, 'belong']] - new_data_start_confidence[data_num]
        temp_value_2 = test_data_end_confidence[new_data.loc[data_num, 'belong']] - new_data_end_confidence[data_num]
        temp_value = (temp_value_2 + temp_value_1) / 2.0
        importance_value.append(temp_value)

    new_data['importance_value'] = importance_value
    df_filtered = new_data[new_data['importance_value'] > -100].reset_index(drop=True)
    # print(len(new_data_df))
    df_filtered.to_csv(os.path.join(data_path, 'temp_data_add_importance_value.csv'), index=False)
    seg_list = []
    seg_score = []
    seg_text = []
    seg_position = []
    for class_num in range(len(test_data)):
        seg_list.append([])
        seg_score.append([])
        seg_text.append([])
        seg_position.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
        seg_text[class_name].append(df_filtered.loc[df_filtered_num, 'belong_text'])
        seg_position[class_name].append(df_filtered.loc[df_filtered_num, 'position'])
    print(seg_list)
    print(seg_score)
    print(seg_text)
    print(seg_position)

    seg_arr = np.array(seg_list)
    seg_score_arr = np.array(seg_score)
    seg_text_arr = np.array(seg_text)
    seg_position_arr = np.array(seg_position)

    np.save(os.path.join(data_path, 'seg.npy'), seg_arr)
    print(len(seg_arr))
    np.save(os.path.join(data_path, 'seg_score.npy'), seg_score_arr)
    print(len(seg_score_arr))
    np.save(os.path.join(data_path, 'seg_text.npy'), seg_text_arr)
    print(len(seg_text_arr))
    np.save(os.path.join(data_path, 'seg_position.npy'), seg_position_arr)
    print(len(seg_position_arr))


if __name__ == '__main__':
    pass
