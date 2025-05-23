from transformers import BertTokenizerFast, BertForQuestionAnswering

def get_model(config):
    model = BertForQuestionAnswering.from_pretrained(config.pretrained_model)
    return model
