# 在与处理中将输入和输出都处理成了仅包含 0 - 1000 之间整数的向量与矩阵，因此可以认为对应的
# transformer模型中，输入和输出的词库大小均为 1000

import numpy as np
import torch
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode
from Model import make_model
from DataGen import data_gen

V = 1000
model = make_model(V, V, N=6)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

# 目标发射的信号、探测器位置、接收点数
detectors = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
tgt_sig = np.zeros(400)
for i in range(10):
    tgt_sig[i::20] = 1
re_num = 100

def run(model, loss, epochs = 10):
    for _ in range(epochs):
        model.train()
        run_epoch(data_gen(detectors, tgt_sig, re_num, 8, 20), model, loss)

        model.eval()
        run_epoch(data_gen(detectors, tgt_sig, re_num, 8, 50), model, loss)

    model.eval()
    d = data_gen(detectors, tgt_sig, re_num, 1, 1)
    src, trg = Variable(d.src), Variable(d.trg)
    src_mask = Variable(torch.ones(1, 1, src.shape[-1]))
    res = greedy_decode(model, src, src_mask, max_len = 3, start_symbol=0)

    return (trg, res)

run(model, loss)