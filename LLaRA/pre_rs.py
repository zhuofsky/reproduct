import json
from torch.utils.data import Dataset, DataLoader
from recommender.A_SASRec_final_bce_llm import SASRec
import torch
import numpy as np
import random
from tqdm import tqdm
import logging

# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
logfile = './log/SASRec_only.txt'
fh = logging.FileHandler(logfile, mode='w')  # open的打开模式这里可以进行参考
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第五步，将logger添加到handler里面
logger.addHandler(fh)

class SASRecDataset(Dataset):
    def __init__(self, datas,item_num,maxlen=10,flag='train'):
        self.datas = datas
        self.item_num = item_num
        self.maxlen = maxlen
        self.seq_datas = []
        self.flag = flag
        self.pos = []
        self.neg = []
        if flag == 'train':
            datas = datas['train']
        elif flag == 'val':
            datas = datas['val']
        else:
            datas = datas['test']
        for data in datas:
            if data['len_seq'] >= 3:
                self.seq_datas.append(data['seq'])
                self.pos.append(data['next'])
                neg_id = random.randint(1,self.item_num)
                while neg_id in data['seq'] or neg_id == data['next']:
                    neg_id = random.randint(1,self.item_num)
                self.neg.append(neg_id)
        self.seq_datas = torch.tensor(self.seq_datas)
        self.pos = torch.tensor(self.pos)
        self.neg = torch.tensor(self.neg)
        assert len(self.seq_datas) == len(self.pos)
        assert len(self.seq_datas) == len(self.neg)

    def __len__(self):
        return len(self.seq_datas)

    def __getitem__(self, idx):
        # pos_list = np.zeros([self.item_num],dtype=np.float32)
        # pos_list[self.pos[idx]-1] = 1
        # pos_list = torch.tensor(pos_list)
        # if self.flag == 'train':
        #     return self.seq_datas[idx], pos_list
        # elif self.flag == 'val':
        #     return self.seq_datas[idx], self.pos[idx]
        # 正例负例
        nxt = self.pos[idx]
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        idxx = self.maxlen - 1
        for i in reversed(self.seq_datas[idx]):
            seq[idxx] = i
            pos[idxx] = nxt
            if nxt != 0:
                neg[idxx] = random.randint(1, self.item_num+1)
                while neg[idxx] in self.seq_datas[idx] or neg[idxx] in pos:
                    neg[idxx] = random.randint(1, self.item_num+1)
            nxt = i
            idxx -= 1
            if idxx == -1:break
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        neg = torch.tensor(neg)
        return seq, pos, neg


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_path = 'D:/pycharm_project/cdr/my_reproduct/LLaRA/data/Movies_and_TV_data/train_test_split_max_length.json'
    datamaps_path = 'D:/pycharm_project/cdr/my_reproduct/LLaRA/data/Movies_and_TV_data/datamaps.json'
    save_path = './rec_model'
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(datamaps_path, 'r') as f:
        datamaps = json.load(f)
    item_num = len(datamaps["id2item"].keys())
    batch_size = 5196
    hidden_size = 64
    state_size = 10
    len_state  = [state_size-1]*batch_size
    dropout = 0.1
    lr = 0.001
    max_epochs = 200
    max_hit_ratio = 0
    model = SASRec(hidden_size, item_num, state_size, dropout, device, num_heads=1).to(device)

    train_dataset = SASRecDataset(data,item_num, flag='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SASRecDataset(data,item_num, flag='val')
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    for epoch in range(max_epochs):
        loss_total = 0
        model.train()
        train_iter = tqdm(enumerate(train_dataloader),
                         desc=f"train: Epoch {epoch}/{max_epochs}",
                         total=len(train_dataloader))
        # for idx, (seq, pos, neg) in train_iter:
        #     seq_data = seq_data.to(device)
        #     pos_logit = pos_logit.to(device)
        #     logit = model(seq_data, len_state)
        #     loss = loss_fn(logit, pos_logit)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     loss_total += loss.item()
        #     train_iter.set_postfix(loss=f"{loss_total/(idx+1):.6f}", lr=lr, batch_size=batch_size)
        # if epoch % 1 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         eval_iter = tqdm(enumerate(eval_dataloader),
        #                           desc=f"val: Epoch {epoch}/{max_epochs}",
        #                           total=len(eval_dataloader))
        #         total_num = len(eval_dataloader) * batch_size
        #         hits_num = 0
        #         for idx, (seq_data, pos_ids) in eval_iter:
        #             seq_data = seq_data.to(device)
        #             logit = model(seq_data, len_state)
        #             pre_ids = torch.argmax(logit, dim=1)
        #             for pre_id, pos_id in zip(pre_ids,pos_ids):
        #                 if pre_id + 1 == pos_id:
        #                     hits_num += 1
        #             eval_iter.set_postfix(hits_num =hits_num, hit_ratio=hits_num/total_num)
        #         hit_ratio = hits_num/total_num
        #         print(f"eval: {epoch}/{max_epochs} 准确率: {hit_ratio}")
        #         if hit_ratio > max_hit_ratio:
        #             torch.save(model.state_dict(), save_path + 'movies_and_tv.pt')
        #             max_hit_ratio = hit_ratio
        for idx, (seq, pos, neg) in train_iter:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits= model(seq, len_state, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), \
                                     torch.zeros(neg_logits.shape, device=device)
            indices = torch.where(pos != 0)
            loss = bce_loss(pos_logits[indices], pos_labels[indices])
            loss += bce_loss(neg_logits[indices], neg_labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            train_iter.set_postfix(loss=f"{loss_total/(idx+1):.6f}", lr=lr, batch_size=batch_size)
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                eval_iter = tqdm(enumerate(eval_dataloader),
                                  desc=f"val: Epoch {epoch}/{max_epochs}",
                                  total=len(eval_dataloader))
                total_num = len(eval_dataloader) * batch_size
                hits_num = 0
                for idx, (seq, pos, neg) in eval_iter:
                    seq_data = seq_data.to(device)
                    logit = model(seq_data, len_state)
                    pre_ids = torch.argmax(logit, dim=1)
                    for pre_id, pos_id in zip(pre_ids,pos_ids):
                        if pre_id + 1 == pos_id:
                            hits_num += 1
                    eval_iter.set_postfix(hits_num =hits_num, hit_ratio=hits_num/total_num)
                hit_ratio = hits_num/total_num
                print(f"eval: {epoch}/{max_epochs} 准确率: {hit_ratio}")
                if hit_ratio > max_hit_ratio:
                    torch.save(model.state_dict(), save_path + 'movies_and_tv.pt')
                    max_hit_ratio = hit_ratio





