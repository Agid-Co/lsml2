from celery import Celery
import functools
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert_base_uncased', return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        output_1, output_2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )

        bert_output = self.bert_drop(output_2)
        output = self.out(bert_output)
        return output


def sentence_prediction(sentence):
    MODEL = BERTBaseUncased()
    DEVICE = 'cpu'
    PREDICTION_DICT = dict()
    MODEL.load_state_dict(torch.load('my_model.bin' , map_location=torch.device('cpu')))
    MODEL.to(DEVICE)
    MODEL.eval()
    
    tokenizer = transformers.BertTokenizer.from_pretrained(
    'bert_base_uncased',
    do_lower_case =True
    )
    max_len = 512
    review = str(sentence)
    review = " ".join(review.split())
    
    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    return outputs[0][0]


celery_app = Celery('tasks', backend='redis://redis', broker='redis://redis')

@celery_app.task(name='tasks.predict')
def predict(data):
    result = sentence_prediction(data).tolist()
    return result
