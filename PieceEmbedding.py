from torch import nn
import math
import torch
from constants import *

class PieceEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PieceEmbedding, self).__init__()
        assert d_model % 4 == 0 # must be divisible twice for positional encoding lookup table to form correctly
        self.d_model = d_model
        vocab_size = len(ALL_PIECES)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        boardLength = 8
        self.register_buffer('pe', create_positional_encoding_lookup(boardLength, d_model))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        tokens = x[:, :, 0]
        first_axis_position = x[:, :, 1]
        second_axis_position = x[:, :, 2]
        token_embeddings = self.token_embedding(tokens) * math.sqrt(self.d_model)
        position_embeddings = self.encode_grid_position(first_axis_position, second_axis_position).requires_grad_(False)
        return token_embeddings + position_embeddings

    def encode_grid_position(self, first_axis_position, second_axis_position):
        first_axis_embedding = self.pe[first_axis_position]
        second_axis_embedding = self.pe[second_axis_position]
        return torch.cat((first_axis_embedding, second_axis_embedding), dim=-1)

def create_positional_encoding_lookup(max_position, d_model):
    positional_encoding = torch.zeros((max_position, d_model // 2))
    position = torch.arange(0, max_position, dtype=torch.float32).unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    return positional_encoding
