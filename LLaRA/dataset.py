import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_json,load_txt
from tqdm import tqdm

class llara_dataset(Dataset):
    def __init__(self, args, stage='train'):
        super(llara_dataset, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.max_len = args.max_len
        self.max_cans_len = args.max_cans_len
        self.read_data(stage)
        self.padding_item_id = int(self.state['item_num']) + 1  # 填充的id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_data(self, stage):
        train_test_split_max_length_file = self.data_dir + '/train_test_split_max_length.json'
        datamaps_file = self.data_dir + '/datamaps.json'  # datamap
        state_file = self.data_dir + '/state.json'
        train_test_split_max_length = load_json(train_test_split_max_length_file)
        self.datamaps = load_json(datamaps_file)
        self.state = load_json(state_file)
        '''
        sample ={
                'user_id': ,
                'seq': ,
                'times': ,
                'ratings': ,
                'len_seq': ,
                'next': ,
                'next_time': ,
                'next_rating': ,
                'unpad_seq':,
                'unpad_time':,
                'unpad_rating':,
            }
        '''
        assert stage in ['train', 'val', 'test']
        if stage == 'train':
            data = train_test_split_max_length['train']
        elif stage == 'val':
            data = train_test_split_max_length['val']
        else:
            data = train_test_split_max_length['test']
        del train_test_split_max_length
        new_data = []
        for sample in data:
            if sample['len_seq'] < 5:
                continue
            sample['seq_name'] = [self.datamaps['itemid2title'][str(item)] for item in sample['unpad_seq']]
            sample['correct_name'] = self.datamaps['itemid2title'][str(sample['next'])]
            sample['cans'] = self.negative_sampling(sample['unpad_seq'],sample['next'])
            random.shuffle(sample['cans'])
            sample['cans_name'] = [self.datamaps['itemid2title'][str(item)] for item in sample['cans']]
            # sample.pop('unpad_times')
            sample.pop('unpad_ratings')
            # sample.pop('times')
            sample.pop('ratings')
            # sample.pop('next_time')
            sample.pop('next_rating')
            new_data.append(sample)
            '''
            sample ={
                    'user_id': ,
                    'seq': ,
                    'seq_name': ,
                    'cans': ,
                    'cans_name': ,
                    'len_seq': ,
                    'next': ,
                    'correct_name': ,
                    'unpad_seq':,
                }
            '''
        del data
        self.data = new_data

    def negative_sampling(self, seq, next):
        max_id = len(self.datamaps['id2item']) + 1
        cans = [next]
        while len(cans) < self.max_cans_len:
            random_id = np.random.randint(1, max_id)
            if random_id not in cans and random_id not in seq and str(random_id) in self.datamaps['itemid2title'].keys():
                cans.append(random_id)
        return cans

class TrainCollater:
    def __init__(self,
                 prompt_path = None,
                 llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = load_txt(prompt_path)
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        instruction  = self.prompt_list
        inputs_text = [instruction] * len(batch)
        thresh_hold = self.cur_step/self.max_step
        p = random.random()  # 这个阈值是为了让有的训练带Embedding，有的不带
        if p < thresh_hold or not self.train:
            for i, sample in enumerate(batch):
                input_text = inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt = ", ".join([seq_title + ' [HistoryEmb]' for seq_title in sample['seq_name']])
                    input_text = input_text.replace('[HistoryHere]', insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt = ", ".join([can_title + ' [CansEmb]' for can_title in sample['cans_name']])
                    input_text = input_text.replace('[CansHere]', insert_prompt)
                inputs_text[i] = input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [PH]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [PH]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1

        targets_text = [sample['correct_name'] for sample in batch]
        if self.train:
            targets_text = [target_text + self.terminator for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]
            batch_tokens = self.llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            new_batch={"tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(len(sample['cans'])) for sample in batch], dim=0),
                       "next": torch.stack([torch.tensor(sample['next']) for sample in batch], dim=0),
                       "flag":flag,
                       }
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            cans_name = [sample['cans_title'] for sample in batch]
            new_batch={"tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(len(sample['cans'])) for sample in batch], dim=0),
                       "next": torch.stack([torch.tensor(sample['next']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       "cans_name": cans_name,
                       }
        return new_batch
