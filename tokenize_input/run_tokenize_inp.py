from tokenize_input.summ_tokenize import SummTokenize
import json
import torch
from tqdm import tqdm

if __name__ == '__main__':
    tokenizer = SummTokenize()

    phase = 'val'
    with open(f'/Users/LongNH/Workspace/presumm-vn/json/json_data.{phase}.json', 'r') as f:
        lis_data = json.load(f)

    for i, data in enumerate(tqdm(lis_data)):
        (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask) \
            , (tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask) = \
            tokenizer.tokenizing_formatted_input(**data, is_pad=True)
        abs_stacked_inp = torch.stack([
            src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask,
            tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask
        ], dim=0)
        torch.save(abs_stacked_inp, f'/Users/LongNH/Workspace/presumm-vn/abs_bert_data/{phase}/abs_bert_data{i}.pt')
