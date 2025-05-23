import torch
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('/media/usr/external/home/usr/project/project3_data/base_model/bert_base_chinese')
class OCNLI_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):

        sentence1 = dataset['premise']
        sentence2 = dataset['hypothese']
        # 用 _ 将前提和假设拼接在一起，但这应该不是好的做法
        sentence1_2 = ['{}_{}'.format(a, b) for a, b in zip(sentence1, sentence2)]
        self.texts = [tokenizer(
            sentence,
            padding='max_length',
            # bert最大可以设置到512，对OCNLI的统计计算中，
            # 发现所有数据没有超过128，max_length越大，计算量越大
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ) for sentence in sentence1_2]
        self.labels = [label for label in dataset['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]