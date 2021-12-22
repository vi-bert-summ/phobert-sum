from torch import nn, Tensor
from config_const import MODEL_DIM, MAX_SEQ_LENGTH
import torch


class AbsDecoder(nn.Module):
    def __init__(self, n_head=8, n_decoder_block=1):
        super(AbsDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(batch_first=True, d_model=MODEL_DIM, nhead=n_head, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_block)
        self.tgt_mask = torch.triu(torch.ones(size=(MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)), diagonal=1).T.bool()

    # tgt: đầu vào của decoder (batch_size * tgt_seq_len * embed_dim)
    # memory: embed của input sequence (batch_size * src_seq_len * embed_dim)
    # tgt_key_padding_mask: mask các token padding của tgt (batch_size * tgt_seq_len)
    # memory_key_padding_mask: mask các token padding của src (batch_size * src_seq_len)
    # tgt_mask: attention mask của decoder (để mô hình không nhìn được token ở tương lai)
    def forward(self, tgt: Tensor, memory: Tensor, tgt_key_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        self.tgt_mask = self.tgt_mask.to(tgt.device)
        out = self.transformer_decoder(tgt=tgt, memory=memory,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       tgt_mask=self.tgt_mask)
        return out
