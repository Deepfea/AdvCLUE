import pandas as pd
import torch
from transformers import BertTokenizer

import os

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    eval_steps = 500
    lr = 1e-6
    epochs = 10
    batch_size = 16
    max_length = 510
    output_model_dir = '/media/usr/external/home/usr/project/project3_data/adv_samples/CMRC2018/bert_base_chinese'
    pretrained_dir = '/media/usr/external/home/usr/project/project3_data/dataset/CMRC2018'
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def get_token(data_df):
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    paragraphs = []
    for text in data_df['context']:
        paragraphs.append(str(text))
    questions = []
    for text in data_df['question']:
        questions.append(str(text))
    start_positions = []
    for text in data_df['answer_start']:
        start_positions.append(text)
    end_positions = []
    for text in data_df['answer_end']:
        end_positions.append(text)
    content = {
        "tokens": list(),
        "input_ids": list(),
        "token_type_ids": list(),
        "mask_ids": list(),
        "start_positions": list(),
        "end_positions": list()
    }

    for num in range(len(paragraphs)):
        tokens = []
        token_type_ids = []
        text_a = paragraphs[num]
        tokens_a = []
        for num_a in range(len(text_a)):
            tokens_a.append(text_a[num_a])
        text_b = questions[num]
        tokens_b = []
        for num_b in range(len(text_b)):
            tokens_b.append(text_b[num_b])

        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)
        tokens.append("[SEP]")
        token_type_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        content['tokens'].append(tokens)
        content['input_ids'].append(input_ids)
        content['token_type_ids'].append(token_type_ids)
        content['mask_ids'].append(input_mask)
        content['start_positions'].append(start_positions[num])
        content['end_positions'].append(end_positions[num])

    assert len(content['tokens']) == len(content['input_ids']) == len(content['token_type_ids']) == len(content['mask_ids']) == len(content['start_positions'])  == len(content['end_positions'])
    return content

if __name__ == "__main__":
    pass
