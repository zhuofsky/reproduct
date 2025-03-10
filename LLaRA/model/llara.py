import torch
import os
from torch import nn
import argparse
import random
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from .mlp_pojector import MlpProjector
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
from SASRecModules_ori import *

class LLaRA(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(LLaRA, self).__init__()
        self.args = args
        self.load_llm()
        self.load_rec_model()
        self.load_projector()

    def load_llm(self):
        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.args.llm_path)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(self.args.llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.args.llm_tuning == 'lora':
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                     inference_mode=False,
                                     r=self.args.lora_r,
                                     lora_alpha=self.args.lora_alpha,
                                     lora_dropout=self.args.lora_dropout,
                                     target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj',
                                                     'down_proj'])
            self.peft_config = peft_config
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.args.llm_tuning == 'freeze':
            # for name, param in self.llama_model.named_parameters():
            #     param.requires_grad = False
            pass
        elif self.args.llm_tuning == 'freeze_lora':
            # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
            #                          inference_mode=False,
            #                          r=self.hparams.lora_r,
            #                          lora_alpha=self.hparams.lora_alpha,
            #                          lora_dropout=self.hparams.lora_dropout,
            #                          target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj',
            #                                          'down_proj'])
            # self.peft_config = peft_config
            # self.llama_model = get_peft_model(self.llama_model, peft_config)
            # for name, param in self.llama_model.named_parameters():
            #     param.requires_grad = False
            pass
        else:
            raise NotImplementedError()
        print('Loading LLAMA Done')

    def load_rec_model(self):
        print('Loading Rec Model')
        self.rec_model = torch.load(self.args.rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def load_projector(self):
        self.projector = MlpProjector(self.args.rec_size, self.llama_model.config.hidden_size)
        print('Loading Projector Done')


    def forward(self, batch):
        targets = batch['tokens'].input_ids.masked_fill(
            batch['tokens'].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:, 1:], -100)
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)

        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()

        his_item_embeds= self.encode_items(batch["seq"])
        cans_item_embeds= self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["next"])

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == his_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["token"].input_ids[i]==his_token_id).nonzero().view(-1)
                # 这个就转化为了一维的向量
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i,-batch["len_seq"][i].item():]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == cans_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, -batch["len_cans"][i].item():]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds

    def encode_items(self, seq):
        if self.args.rec_embed == "SASRec":
            item_rec_embs = self.rec_model.cacu_x(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs

    def generator(self,
                  batch,
                  temperature=0.8,
                  do_sample=False,
                  num_beams=1,
                  max_gen_length=64,
                  min_gen_length=1,
                  repetition_penalty=1.0,
                  length_penalty=1.0,
                  num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
            )
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs