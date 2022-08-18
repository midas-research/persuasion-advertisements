import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel

import numpy as np

class BERTEncoder(nn.Module):
    def __init__(self, config, is_training = True):
        super(BERTEncoder, self).__init__()
        self.enc =  BertModel.from_pretrained('bert-base-uncased')
        self.att = nn.Linear(768, 1)
        self.fc = nn.Linear(768, config['embedding_size'])

    def forward(self, text_strings, text_lengths):
        src = self.enc(text_strings)
        src = src.last_hidden_state
        wt = self.att(src)
        # wt = [batch size, src len, 1]
        wt = torch.softmax(wt, dim=1)
        src = torch.matmul(wt.permute(0, 2, 1), src)
        # src = [batch size, 1, 768]
        src = self.fc(src)
        # src = [batch size, 1, embed_size]
        src = torch.squeeze(src, dim=1)
        # src = [batch size, embed_size]
        return src
