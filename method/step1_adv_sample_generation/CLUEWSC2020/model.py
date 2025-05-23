import os
from transformers import BertModel
import torch
from torch import nn


class BERT(nn.Module):
    __name__ = "BERT"

    def __init__(self, config):
        super(BERT, self).__init__()
        self.embeddings = BertModel.from_pretrained(config.pretrained_model)
        self.device = config.device

    def forward(self, x):
        input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end = x
        embedded = self.embeddings(input_ids, attention_mask)
        indexs = torch.cat(
            (
                span1_begin.unsqueeze(1),
                span1_end.unsqueeze(1) - 1,
                span2_begin.unsqueeze(1),
                span2_end.unsqueeze(1) - 1,
            ),
            1,
        )
        indexs = indexs.unsqueeze(2).repeat(1, 1, embedded[0].shape[2])
        span_vecs = torch.gather(embedded[0], dim=1, index=indexs)
        cosine = torch.cosine_similarity(
            (span_vecs[:, 0, :] + span_vecs[:, 1, :]) / 2,
            (span_vecs[:, 2, :] + span_vecs[:, 3, :]) / 2,
            dim=1,
        )
        output = torch.cat(tuple([1 - cosine.unsqueeze(1), cosine.unsqueeze(1)]), 1)
        return nn.functional.softmax(output, dim=1)


if __name__ == "__main__":
    pass
