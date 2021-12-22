from models.abs_bert_summ import AbsBertSumm
from tokenize_input.summ_tokenize import SummTokenize
import json
import torch

if __name__ == '__main__':
    tokenizer = SummTokenize()
    print(tokenizer.phobert_tokenizer.vocab_size)
    vocab_size = tokenizer.phobert_tokenizer.vocab_size

    with open('/Users/LongNH/Workspace/presumm-vn/json_data/json_data.train.json', 'r') as f:
        data = json.load(f)
    (src_inp_ids, src_tok_type_ids,
     src_lis_cls_pos, src_mask), (tgt_inp_ids, tgt_tok_type_ids,
                                  tgt_lis_cls_pos, tgt_mask) = \
        tokenizer.tokenizing_formatted_input(**data[0], is_pad=True)

    abs_bert_summ = AbsBertSumm(vocab_size=vocab_size)
    print(src_lis_cls_pos)
    print(abs_bert_summ(src_ids=torch.tensor(src_inp_ids).reshape(1, -1),
                        src_pad_mask=torch.tensor(src_mask).reshape(1, -1),
                        src_token_type=torch.tensor(src_tok_type_ids).reshape(1, -1),
                        # is_freeze_phase1=True,
                        src_cls_pos=torch.tensor(src_lis_cls_pos).reshape(1, -1),
                        tgt_ids=torch.tensor(tgt_inp_ids).reshape(1, -1),
                        tgt_pad_mask=torch.tensor(tgt_mask).reshape(1, -1),
                        tgt_token_type=torch.tensor(tgt_tok_type_ids).reshape(1, -1)).shape),
