import numpy as np
import torch
from torch import nn
from config import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)      # embedding_dim = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)     # embedding (hidden_size) -> hidden_dim (hidden_size)

    def forward(self, inp, hidden):
        embedded = self.embedding(inp).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=Config.device)
