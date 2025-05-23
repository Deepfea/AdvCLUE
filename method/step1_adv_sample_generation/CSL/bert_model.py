import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification, BertTokenizer, BertForPreTraining
from torch.nn import functional as F



class BertClassifier(nn.Module):
    def __init__(self, config, transformer_width=768, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert_layer = BertForSequenceClassification.from_pretrained(config.pretrained_model, return_dict=True)
        self.config = config

    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.config.device)
        token_type_ids = batch["token_type_ids"].to(self.config.device)
        mask_ids = batch["mask_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        self.bert_layer = self.bert_layer.to(self.config.device)
        outputs = self.bert_layer(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids)

        labels = torch.squeeze(labels, dim=-1)

        loss = F.cross_entropy(outputs.logits, labels)
        prediction = outputs.logits.argmax(-1)
        output = outputs.logits.detach()
        return loss, prediction, output

def test():
    bert_model_path = '../checkpoints/bert-base-chinese/' # pytorch_model.bin
    bert_config_path = '../checkpoints/bert-base-chinese/' # bert_config.json
    vocab_path = '../checkpoints/bert-base-chinese/vocab.txt'

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    # model = BertModel.from_pretrained(bert_model_path, config=bert_config_path)
    model = BertForPreTraining.from_pretrained(bert_model_path, config=bert_config_path)

    text_batch = ["哈哈哈", "嘿嘿嘿", "嘿嘿嘿", "嘿嘿嘿"]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    print(input_ids)
    print(input_ids.shape)
    output1,output2 = model(input_ids)
    print(output1)
    print(output2)
    print(output1.shape)
    print(output2.shape)


if __name__ == '__main__':
    pass


