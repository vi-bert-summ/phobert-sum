from torch.utils.data import DataLoader

from data.summ_dataset import SummDataset
from models.abs_bert_summ import AbsBertSumm
from models.abs_bert_summ_pylight import AbsBertSummPylight
from tokenize_input.summ_tokenize import SummTokenize
from trainer.trainer_builder import start_training
import argparse


def args_parser():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('-jsondat', '--json_data', required=True, help='Input in json format')
    ap.add_argument('-gpus', '--gpus', required=False, help='Specify gpus device')
    ap.add_argument('-phase', '--phase', required=False, help='Specify phase [train, val, test]')
    ap.add_argument('-batch_size', '--batch_size', required=False, help='Specify the batch size')
    ap.add_argument('-save_ckpt_path', '--save_ckpt_path', required=False, help='Specify the checkpoint path')
    args = vars(ap.parse_args())

    return args


if __name__ == '__main__':
    cmd_args = args_parser()

    train_dataset = SummDataset(bert_data_folder_path=cmd_args.get('json_data'), phase=cmd_args.get('phase'))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(cmd_args.get('batch_size')), shuffle=True)

    tokenizer = SummTokenize()
    vocab_size = tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ_pylight = AbsBertSummPylight(vocab_size=vocab_size)

    val_dataset = SummDataset(bert_data_folder_path=cmd_args.get('json_data'), phase='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)

    start_training(abs_bert_summ_model=abs_bert_summ_pylight, train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader, gpus=cmd_args.get('gpus'),
                   save_ckpt_path=cmd_args.get('save_ckpt_path')
                   )
