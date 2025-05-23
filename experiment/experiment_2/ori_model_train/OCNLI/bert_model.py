import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertForPreTraining
from torch.nn import functional as F

class BertClassifier(nn.Module):
    def __init__(self, config, transformer_width=768, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert_layer = BertForSequenceClassification.from_pretrained(config.pretrained_model, num_labels=3, return_dict=True)
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


if __name__ == '__main__':
    pass


