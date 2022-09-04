import torch
from torch import nn

from lib.transformer import GateControlTransformer
from lib.position_embedding import positionEmbeddingLearned


class cross_scale_transformer(nn.Module):
    def __init__(self,
                 num_queries=1,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 return_intermediate_dec=False):

        super().__init__()

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.pos_embed = positionEmbeddingLearned(d_model // 2)
        self.num_queries = num_queries
        self.transformer = GateControlTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            return_intermediate_dec=return_intermediate_dec)
    def forward(self, x):
        pos_embed = self.pos_embed(x).to(x.dtype)
        o, c = self.transformer(
            x, None, self.query_embed.weight, pos_embed)

        return o, c