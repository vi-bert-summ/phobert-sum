import json
import torch

from tokenize_input.summ_tokenize import SummTokenize
THRESHOLD = 0.3


def summarize(input_fp, output_fp, model):
    tokenizer = SummTokenize()

    with open(input_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

    (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask), lis_tgt = \
        tokenizer.tokenizing_ext_input(**data, is_pad=True)
    src_inp_ids = torch.unsqueeze(src_inp_ids, 0)
    src_tok_type_ids = torch.unsqueeze(src_tok_type_ids, 0)
    src_lis_cls_pos = torch.unsqueeze(src_lis_cls_pos, 0)
    src_mask = torch.unsqueeze(src_mask, 0)
    masked_out_prob = model.predict_step(batch=[src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask, None],
                                         batch_idx=0)
    masked_out_prob = masked_out_prob.reshape(-1)
    src_lis_tok = data.get('src')

    res = ''
    for ids in range(len(src_lis_tok)):
        if masked_out_prob[ids] >= THRESHOLD:
            res += ' '.join(src_lis_tok[ids])
            
    with open(output_fp, 'w', encoding="utf-8") as file:
        file.write(res)

    return res


# if __name__ == '__main__':
#     ext_bert_summ_pylight = ExtBertSummPylight()
#     inference_json(json_path='/Users/LongNH/Workspace/presumm-vn/inferences/test_sample.json',
#                    model=ext_bert_summ_pylight)
