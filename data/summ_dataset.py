from torch.utils.data import Dataset
import glob
import os
import torch


class SummDataset(Dataset):
    def __init__(self, bert_data_folder_path, phase='train'):
        self.data_folder_path = os.path.join(bert_data_folder_path, phase, '*.pt')
        self.list_file_path = glob.glob(self.data_folder_path)

    def __len__(self):
        return len(self.list_file_path)

    def __getitem__(self, idx):
        file_path = self.list_file_path[idx]
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask\
            , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask = torch.load(file_path)
        return (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
                , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask)
