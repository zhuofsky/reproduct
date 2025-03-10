import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from dataset import llara_dataset, TrainCollater
from model.llara import LLaRA
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
from SASRecModules_ori import *
from trainer import llara_Trainer

def llara_dataloader(dataset,model, args, stage='train'):
    if stage == 'train':
        dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=TrainCollater(prompt_path=args.prompt_path,
                                                               llm_tokenizer=model.llama_tokenizer,
                                                               train=True,
                                                               max_step=args.max_epochs * (
                                                                       len(dataset) // args.batch_size) // args.num_workers))
    else:
        dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    collate_fn=TrainCollater(prompt_path=args.prompt_path,
                                                             llm_tokenizer=model.llama_tokenizer,
                                                             train=False,
                                                             max_step=args.max_epochs * (
                                                                         len(dataset) // args.batch_size) // args.num_workers))
    return dataloader


def main(args):
    assert args.mode in ['train', 'test']
    model = LLaRA(args)
    train_dataset = llara_dataset(args, stage='train')
    train_dataloader = llara_dataloader(train_dataset,model,args,stage='train')
    val_dataset = llara_dataset(args, stage='val')
    val_dataloader = llara_dataloader(val_dataset, model, args, stage='val')
    test_dataset = llara_dataset(args, stage='test')
    test_dataloader = llara_dataloader(test_dataset, model, args, stage='test')
    trainer = llara_Trainer(model,model.llama_tokenizer,
                            train_dataloader,
                            val_dataloader,
                            test_dataloader,
                            args)
    if args.mode == 'train':
        for epoch in range(args.max_epochs):
           trainer.pretrain(epoch,train_dataloader, mode='train')

    elif args.mode == 'test':
        pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 初始化常见参数
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    # 模型设置参数
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--lr', default=8e-4, type=float)
    parser.add_argument('--mode', default='train', type=str,help='[train,test]')
    parser.add_argument('--weight_decay', default=1e-5, type=float)


    # warm_lr
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)

    # 数据集参数
    parser.add_argument('--dataset', default='Movies_and_TV', type=str)
    parser.add_argument('--data_dir', default='./data/Movies_and_TV_data', type=str)
    parser.add_argument('--llm_path', default="D:/pycharm_project/llm_base/llama3_1-8B-instruct/", type=str)
    parser.add_argument('--prompt_path', default='./prompt/movie.txt', type=str)
    parser.add_argument('--rec_model_path', default='D:/pycharm_project/cdr/my_reproduct/LLaRA/rec_model/movielens.pt', type=str)
    parser.add_argument('--max_len', default=10, type=int, help='seqs的max_len')
    parser.add_argument('--max_cans_len', default=20, type=int)
    parser.add_argument('--rec_size', default=64, type=int)

    # finetuning
    parser.add_argument('--llm_tuning', default='lora', choices=['lora', 'freeze', 'freeze_lora'], type=str)
    parser.add_argument('--peft_dir', default=None, type=str)
    parser.add_argument('--peft_config', default=None, type=str)
    parser.add_argument('--lora_r', default=8, type=float)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    # 其他
    parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser', 'GRU'], type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)

    # parser.add_argument('--accelerator', default='gpu', type=str)
    # parser.add_argument('--devices', default=-1, type=int)
    # parser.add_argument('--precision', default='bf16', type=str)
    # parser.add_argument('--amp_backend', default="native", type=str)
    #
    # parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--num_workers', default=8, type=int)
    # parser.add_argument('--seed', default=1234, type=int)
    # parser.add_argument('--lr', default=1e-3, type=float)
    # parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    # parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    #
    # parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    # parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    # parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    #
    # parser.add_argument('--load_best', action='store_true')
    # parser.add_argument('--load_dir', default=None, type=str)
    # parser.add_argument('--load_ver', default=None, type=str)
    # parser.add_argument('--load_v_num', default=None, type=int)
    #
    # parser.add_argument('--dataset', default='movielens_data', type=str)
    # parser.add_argument('--data_dir', default='data/ref/movielens', type=str)
    # parser.add_argument('--model_name', default='mlp_projector', type=str)
    # parser.add_argument('--loss', default='lm', type=str)
    # parser.add_argument('--weight_decay', default=1e-5, type=float)
    # parser.add_argument('--no_augment', action='store_true')
    # parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    # parser.add_argument('--log_dir', default='movielens_logs', type=str)
    #
    # parser.add_argument('--rec_size', default=64, type=int)
    # parser.add_argument('--padding_item_id', default=1682, type=int)
    # parser.add_argument('--llm_path', default="D:/pycharm_project/model/llama2/models_hf/7B", type=str)
    # parser.add_argument('--rec_model_path', default='./rec_model/movielens.pt', type=str)
    # parser.add_argument('--prompt_path', default='./prompt/movie/', type=str)
    # parser.add_argument('--output_dir', default='./output/', type=str)
    # parser.add_argument('--ckpt_path', type=str)
    # parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser', 'GRU'], type=str)
    #
    # parser.add_argument('--aug_prob', default=0.5, type=float)
    # parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    # parser.add_argument('--auto_lr_find', default=False, action='store_true')
    # parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    # parser.add_argument('--max_epochs', default=10, type=int)
    # parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    # parser.add_argument('--cans_num', default=10, type=int)


    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main(args)
