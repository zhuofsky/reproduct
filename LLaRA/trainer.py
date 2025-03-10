import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import load_txt
from optims import LinearWarmupCosineLRScheduler


class llara_Trainer:
    def __init__(self,
                 model,
                 llm_tokenizer,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 args):
        self.args = args
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.llm_tokenizer = llm_tokenizer
        # 定义优化器
        self.steps_per_epoch = len(train_dataloader) // args.num_workers
        self.max_steps = args.max_epochs * self.steps_per_epoch
        self.optimizer = self.get_optimizer()
        # self.loss_fc = torch.nn.CrossEntropyLoss() 这里用不上，因为是LLM自带的损失
        self.instruction = load_txt(args.prompt_path)

    def get_optimizer(self):
        if self.args.weight_decay != 0:
            weight_decay = self.args.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.model.projector.parameters(), 'lr': self.args.lr, 'weight_decay': weight_decay},
            {'params': self.model.llama_model.parameters(), 'lr': self.args.lr},
        ])
        if self.args.lr_scheduler is None:
            return optimizer
        else:
            warmup_steps = self.max_steps // 20
            print(f'max_step: {self.max_steps}')
            print(f'warmup_steps: {warmup_steps}')
            if self.args.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                               max_step = self.max_steps,
                                                               min_lr= self.args.lr_decay_min_lr,
                                                               init_lr= self.args.lr,
                                                               warmup_steps=warmup_steps,
                                                               warmup_start_lr=self.args.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def pretrain(self,epoch, data_loader, mode='train'):
        rec_data_iter = tqdm(enumerate(data_loader),
                             desc = f"{mode}: Epoch {epoch}/{self.args.max_epochs}",
                             total=len(data_loader))
        if mode == 'train':
            self.model.train()
            loss_llm = 0 # 大模型的损失
            for i, batch in rec_data_iter:
                cur_step = self.steps_per_epoch * epoch + i
                if self.scheduler:
                    self.scheduler.step(cur_step, epoch, self.max_steps)
                if batch["flag"]:
                    for name, param in self.model.projector.named_parameters():
                        param.requires_grad = False
                else:
                    for name, param in self.model.projector.named_parameters():
                        param.requires_grad = True
                out = self.model(batch)

                if self.args.loss == 'lm':
                    loss = out.loss
                else:
                    raise ValueError("Invalid Loss Type!")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_llm += loss.detach().item()
                print(f'loss: {loss_llm/(i+1)}')
        else:
            self.model.eval()
            val_content = {
                "generate": [],
                "real": [],
                "cans": [],
            }
            for i, batch in rec_data_iter:
                generate_output = self.model.generate(batch)
                for idx, generate in enumerate(generate_output):
                    real = batch['correct_name'][idx]
                    cans = batch['cans_name'][idx]
                    generate = generate.strip().split("\n")[0]
                    val_content["generate"].append(generate)
                    val_content["real"].append(real)
                    val_content["cans"].append(cans)
            # 将val_content 保存至文件中
            self.save_val_content(val_content)

    def calculate_hr1(self, eval_content):
        correct_num = 0
        valid_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1

            generate=generate.strip().lower().strip()
            real=real.strip().lower().strip()
            cans=[item.strip().lower().strip() for item in cans]

            gen_cans_list = []
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)

            if len(gen_cans_list)==1:
                valid_num+=1
                if real == gen_cans_list[0]:
                    correct_num+=1
        valid_ratio = valid_num / total_num
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio, hr1

    def save_val_content(self, eval_content):
        df = pd.DataFrame(eval_content)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        df.to_csv(os.path.join(self.args.output_dir, 'val.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(eval_content)
        metric = hr * prediction_valid_ratio
        print(prediction_valid_ratio,hr,metric)



