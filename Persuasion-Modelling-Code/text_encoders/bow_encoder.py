import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

class BOWEncoder(nn.Module):
    def __init__(self, config, is_training = True):
        super(BOWEncoder, self).__init__()
        '''
        filename = config['init_emb_matrix_path']
        with open(filename, 'rb') as fp:
            word2vec = np.load(fp)
        '''

        #Not using any pretrained embeddings for now:
        
        #self.embedding_weights = torch.from_numpy(word2vec)
        #self.embedding_weights = torch.Tensor(word2vec)
        #self.embedding_layer = nn.Embedding.from_pretrained(self.embedding_weights)
        
        self.embedding_layer = nn.Embedding(config['vocab_size'], config['embedding_size'])

        self.dropout = nn.Dropout(1-config['dropout_keep_prob'])

    def forward(self, text_strings, text_lengths):
        embeddings = self.embedding_layer(text_strings)
        embeddings = self.dropout(embeddings)

        max_text_len = list(text_strings.size())[1]

        boolean_mask = torch.less(torch.arange(max_text_len, dtype=torch.int32).to(text_strings.device), torch.unsqueeze(text_lengths.type(torch.int32), 1))
        weights = boolean_mask.type(torch.float32)
        weights = torch.div(weights, torch.maximum(torch.tensor(1e-12), torch.tile(torch.unsqueeze(text_lengths.type(torch.float32), 1), (1, max_text_len))))

        text_encoded = torch.squeeze(torch.matmul(torch.unsqueeze(weights, 1), embeddings), 1)

        return text_encoded #embeddings, embeddings
