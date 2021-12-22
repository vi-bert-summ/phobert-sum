import torch
from torch import nn
from torch.nn import Sequential

from config_const import CACHED_MODEL_PATH, MODEL_DIM
from models.ext_decoder import ExtDecoder
from models.phobert_model import PhoBert
from models.utils import get_cls_embed, padding_and_stack_cls


class ExtBertSumm(nn.Module):
    def __init__(self):
        super(ExtBertSumm, self).__init__()
        self.phase1_bert = PhoBert(large=False, temp_dir=CACHED_MODEL_PATH, is_freeze=False)
        self.phase2_decoder = ExtDecoder()
        self.sent_classifier = Sequential(
            nn.Linear(MODEL_DIM, 1)
        )

    def forward(self, src_ids, src_pad_mask, src_token_type, src_cls_pos):
        """
        :param src_ids: embeding token của src  (batch_size * seq_len)
        :param src_pad_mask: đánh dấu padding của src (để attention không tính đến nó nữa) (batch_size * seq_len)
        :param src_token_type: đánh dấu đoạn A và B của câu src (batch_size * seq_len)
        :param src_cls_pos: vị trí của các token cls trong src seq (batch_size * num_cls)
        :return:
        """
        # n_batch * n_tokens * n_embed_dim
        embed_phase1 = self.phase1_bert(input_ids=src_ids,
                                        token_type_ids=src_token_type,
                                        attention_mask=src_pad_mask)

        cls_embed = get_cls_embed(tok_embed=embed_phase1, cls_pos=src_cls_pos)  # n_batch * n_cls * n_embed
        padded_cls_embed, pad_mask = padding_and_stack_cls(cls_embed)  # n_batch * MAX_SEQ_LENGTH * n_embed

        out = self.phase2_decoder(src_ids=padded_cls_embed, src_mask=pad_mask)

        logits = self.sent_classifier(out)
        return logits


if __name__ == '__main__':
    ext_bert_summ = ExtBertSumm()
    input_src_ids = torch.randint(size=(1, 512), low=0, high=1000)
    input_src_pad_mask = torch.randint(size=(1, 512), low=0, high=2)
    input_src_token_type = torch.randint(size=(1, 512), low=0, high=2)
    input_src_cls_pos = torch.randint(size=(1, 512), low=0, high=512)
    out_logits = ext_bert_summ(src_ids=input_src_ids, src_pad_mask=input_src_pad_mask,
                               src_token_type=input_src_token_type,
                               src_cls_pos=input_src_cls_pos)
    print(torch.sigmoid(out_logits))
