
from data.summ_dataset import SummDataset
from torch.utils.data import DataLoader

from models.abs_bert_summ import AbsBertSumm


from tokenize_input.summ_tokenize import SummTokenize

if __name__ == '__main__':
    ds = SummDataset(bert_data_folder_path='/Users/LongNH/Workspace/data/abs_bert_data', phase='train')
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True)
    tokenizer = SummTokenize()
    vocab_size = tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ = AbsBertSumm(vocab_size=vocab_size)

    for (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
         , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask) in dl:
        logits = abs_bert_summ(src_ids=src_inp_ids,
                               src_pad_mask=src_mask,
                               src_token_type=src_tok_type_ids,
                               # is_freeze_phase1=True,
                               src_cls_pos=src_lis_cls_pos,
                               tgt_ids=tgt_inp_ids,
                               tgt_pad_mask=tgt_mask,
                               tgt_token_type=tgt_tok_type_ids)
        break
