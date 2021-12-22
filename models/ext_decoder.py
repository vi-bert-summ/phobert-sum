from torch import nn, Tensor
from config_const import MODEL_DIM, MAX_SEQ_LENGTH
import torch


class ExtDecoder(nn.Module):
    def __init__(self, n_head=8, n_encoder_block=6):
        super(ExtDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=MODEL_DIM, nhead=n_head, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_block)

    def forward(self, src_ids, src_mask):
        """
        :param src_ids: (n_batch * MAX_SEQ_LENGTH * n_embed) Embed Cls của BERT phase 1
        :param src_mask: (n_batch * MAX_SEQ_LENGTH * n_embed) Mask của đầu vào
        """
        out = self.transformer_encoder(src=src_ids, src_key_padding_mask=src_mask)
        return out
