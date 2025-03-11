import json
from torch.utils.data import Dataset, DataLoader
from recommender.A_SASRec_final_bce_llm import SASRec
import torch
import numpy as np
import random
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import logging

# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
logfile = './logs/SASRec_only.txt'
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
        self.seq_datas = []
        if flag == 'train':
            train_datas = datas['train']
            for user in train_datas.keys():
                seq_,_,_ = train_datas[user]
                if len(seq_)>=3:
                    self.pos.append(seq_[-1])
                    self.seq_datas.append(seq_[:-1])
            print(f'train 用户总数: {len(self.seq_datas)} ')
        elif flag == 'val':
            train_datas = datas['train']
            valid_datas = datas['val']
            for user in valid_datas.keys():
                pos_id,_,_ = valid_datas[user][0]
                seq_,_,_ = train_datas[user]
                if len(seq_)>=3:
                    self.pos.append(pos_id)
                    self.seq_datas.append(seq_)
            print(f'val 用户总数: {len(self.seq_datas)} ')

    def __len__(self):
        return len(self.seq_datas)

    def __getitem__(self, idx):
        if self.flag == 'train':
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
                    neg[idxx] = np.random.randint(1, self.item_num+1)
                    while neg[idxx] in self.seq_datas[idx] or neg[idxx] in pos:
                        neg[idxx] = np.random.randint(1, self.item_num+1)
                nxt = i
                idxx -= 1
                if idxx == -1:break
            seq = torch.tensor(seq)
            pos = torch.tensor(pos)
            neg = torch.tensor(neg)
            return seq, pos, neg
        elif self.flag == 'val':
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idxx = self.maxlen - 1
            indices = [self.pos[idx]]
            for i in reversed(self.seq_datas[idx]):
                seq[idxx] = i
                idxx -= 1
                if idxx == -1:break
            for _ in range(100):
                t = np.random.randint(1, self.item_num+1)
                while t in self.seq_datas[idx] or t == indices[0]:
                    t = np.random.randint(1, self.item_num + 1)
                indices.append(t)
            return torch.tensor(seq), torch.tensor(indices)


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    data_path = 'D:/pycharm_project/cdr/my_reproduct/LLaRA/data/Movies_and_TV_data/train_test_split.json'
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
    for name, param in model.named_parameters():  # xavier初始化参数
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    train_dataset = SASRecDataset(data,item_num, flag='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SASRecDataset(data,item_num, flag='val')
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    for epoch in range(max_epochs):
        loss_total = 0
        model.train()
        train_iter = tqdm(enumerate(train_dataloader),
                         desc=f"train: Epoch {epoch}/{max_epochs}",
                         total=len(train_dataloader))
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
                # total_num = len(eval_dataloader) * batch_size
                # hits_num = 0
                # for idx, (seq, pos, neg) in eval_iter:
                #     seq_data = seq_data.to(device)
                #     logit = model(seq_data, len_state)
                #     pre_ids = torch.argmax(logit, dim=1)
                #     for pre_id, pos_id in zip(pre_ids,pos_ids):
                #         if pre_id + 1 == pos_id:
                #             hits_num += 1
                #     eval_iter.set_postfix(hits_num =hits_num, hit_ratio=hits_num/total_num)
                # hit_ratio = hits_num/total_num
                # print(f"eval: {epoch}/{max_epochs} 准确率: {hit_ratio}")
                # if hit_ratio > max_hit_ratio:
                #     torch.save(model.state_dict(), save_path + 'movies_and_tv.pt')
                #     max_hit_ratio = hit_ratio
                NDCG = 0
                HT = 0
                max_ht = 0
                val_user = len(eval_dataloader) * batch_size
                for idx, (seq, indices) in eval_iter:
                    seq, indices = seq.to(device), indices.to(device)
                    predictions = -model.predict(seq,indices)
                    ranks = predictions.argsort().argsort()[:, 0]
                    for rank in ranks:
                        if rank <= 10:  # TOP10才记录，这里真实rank = rank + 1 ，因为argsort()索引包含0
                            NDCG += 1 / np.log2(rank + 2)
                            HT += 1
                eval_iter.set_postfix(NDCG=f"{NDCG / (val_user):.4f}", HT=f"{HT / (val_user):.4f}")
                logger.info(f"NDCG@10: {NDCG / val_user:.4f}, Hit@10: {HT / (val_user):.4f}")
                if (HT / (val_user)) > max_ht:
                    torch.save(model.state_dict(), save_path + 'movies_and_tv.pt')
                    max_ht = HT / (val_user)




