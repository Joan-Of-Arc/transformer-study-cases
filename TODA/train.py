import numpy as np
import torch
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode
from Model import make_model
from DataGen import data_gen
import time

V = 1100
model = make_model(V, V, N=6)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

detectors = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
tgt_sig = np.zeros(400)
for i in range(10):
    tgt_sig[i::20] = 1
re_num = 100

def train(batch_size, batch_num):
    model.train()
    run_epoch(
        data_gen(detectors, tgt_sig, re_num, batch_size, batch_num),
        model,
        loss )

    