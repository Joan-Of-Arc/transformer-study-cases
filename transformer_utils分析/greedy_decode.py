import torch
from torch import nn
from train_utils import greedy_decode
from model import Net

model = Net(10, 6)
model.load_state_dict(torch.load("./models_pth/test1.pth"))

src = torch.randint(1, 10, (1, 10))
src[:, 0] = 1
print(src)
out = greedy_decode(model, 2, src, 10, 1)
print(out)