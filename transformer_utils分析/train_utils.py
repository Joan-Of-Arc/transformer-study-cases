# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from torch.nn.modules import Transformer

def get_mask(tgt, head):
    seq_mask = torch.unsqueeze(Transformer.generate_square_subsequent_mask(tgt.size(-1)), 0)
    mask = torch.clone(seq_mask)
    for _ in range(tgt.size(0) * head  - 1):
        mask = torch.concat((mask, torch.clone(seq_mask)))
    return mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, head, pad=0):
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
        self.trg_float = trg.to(torch.float32)
        # 获得 sequence mask
        self.trg_mask = get_mask(trg, head)
        self.ntokens = (self.trg != pad).data.sum()
    

def greedy_decode(model, head, src, max_len, start_symbol):
    memory = model.transformer.encoder(model.embd(src))
    res = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len-1):
        out = model.transformer.decoder(memory, model.embd(res), get_mask(src, head))
        nxt = model.proj(out)
        print(nxt)
        res = torch.cat([res, torch.ones(1, 1).type_as(src.data).fill_(nxt)], dim=1)
    return res

