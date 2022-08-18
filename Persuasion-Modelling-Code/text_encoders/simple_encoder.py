import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

class SimEncoder(nn.Module):
    def __init__(self, config, is_training = True):
        super(SimEncoder, self).__init__()
        
        self.fc1 = nn.Linear(768, 1536)
        self.batch_norm = nn.BatchNorm1d(1536)
        self.dropout = nn.Dropout(1 - config["dropout_keep_prob"])
        self.fc2 = nn.Linear(1536, config['embedding_size'])

    def forward(self, text_strings, text_lengths):
        src = self.dropout(self.batch_norm(self.fc1(text_strings)))
        # src = [batch size, 1536]
        src = self.fc2(src)
        # src = [batch size, 768]
        return src