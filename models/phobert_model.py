from torch import nn
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW
import torch

from config_const import BERT_LARGE_MODEL, BERT_BASE_MODEL
from torch.nn.functional import mse_loss


class PhoBert(nn.Module):
    """
    Đầu vào: (batch_size, sequence_len)
    sequence_len <= 258.
    """

    def __init__(self, large, temp_dir, is_freeze):
        super(PhoBert, self).__init__()
        # Lựa chọn mô hình BERT-large hoặc BERT-base
        if large:
            self.model = BertModel.from_pretrained(BERT_LARGE_MODEL, cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained(BERT_BASE_MODEL, cache_dir=temp_dir)

        self.config = self.model.config
        self.is_freeze = is_freeze

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        :param input_ids:  embed đầu vào (n_batch * n_seq)
        :param token_type_ids: đánh dấu tok nào thuộc đoạn nào (n_batch * n_seq)
        :param attention_mask: đánh dấu đâu là padding token (n_batch * n_seq)
        :param is_freeze: có train weight của mô hình hay không
        :return:
        """
        # print(input_ids, token_type_ids, attention_mask)
        if not self.is_freeze:
            _ = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)
        else:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)

        return _[0]
