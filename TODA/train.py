import numpy as np
import torch
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode
from Model import make_model
from DataGen import data_gen
import time

V = 1100
lr = 5.0
model = make_model(V, V, N=6)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.95)

detectors = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
tgt_sig = np.zeros(400)
for i in range(10):
    tgt_sig[i::20] = 1
re_num = 100

    