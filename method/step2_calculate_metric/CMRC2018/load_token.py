import torch
from transformers import BertTokenizerFast
import os

class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    eval_steps = 500
    lr = 1e-6
    epochs = 10
    batch_size = 16
    max_length = 510
    pretrained_model = "/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese"

def get_token(data_df, path):
    tokenizer = BertTokenizerFast.from_pretrained(path)
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
    train_encodings = tokenizer(paragraphs, questions, return_tensors="pt", padding=True, truncation=True,
                                max_length=510, )
    train_encodings["start_positions"] = torch.tensor(
        [
            train_encodings.char_to_token(idx, x)
            if train_encodings.char_to_token(idx, x) != None
            else -1
            for idx, x in enumerate(start_positions)
        ]
    )
    train_encodings["end_positions"] = torch.tensor(
        [
            train_encodings.char_to_token(idx, x - 1)
            if train_encodings.char_to_token(idx, x - 1) != None
            else -1
            for idx, x in enumerate(end_positions)
        ]
    )
    return train_encodings

def get_ori_token(data_df, path):
    tokenizer = BertTokenizerFast.from_pretrained(path)
    paragraphs = []
    for text in data_df['ori_context']:
        paragraphs.append(str(text))
    questions = []
    for text in data_df['ori_question']:
        questions.append(str(text))
    start_positions = []
    for text in data_df['ori_answer_start']:
        start_positions.append(text)
    end_positions = []
    for text in data_df['ori_answer_end']:
        end_positions.append(text)
    train_encodings = tokenizer(paragraphs, questions, return_tensors="pt", padding=True, truncation=True,
                                max_length=510, )
    train_encodings["start_positions"] = torch.tensor(
        [
            train_encodings.char_to_token(idx, x)
            if train_encodings.char_to_token(idx, x) != None
            else -1
            for idx, x in enumerate(start_positions)
        ]
    )
    train_encodings["end_positions"] = torch.tensor(
        [
            train_encodings.char_to_token(idx, x - 1)
            if train_encodings.char_to_token(idx, x - 1) != None
            else -1
            for idx, x in enumerate(end_positions)
        ]
    )
    return train_encodings