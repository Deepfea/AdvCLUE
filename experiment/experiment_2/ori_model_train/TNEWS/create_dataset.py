from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df, tokenizer):
        max_length = 256
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length=max_length,  # 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)