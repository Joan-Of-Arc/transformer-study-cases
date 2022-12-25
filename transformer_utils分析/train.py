import torch 
from model import Net
from torch import nn 
from train_utils import *

net = Net(10, 6)  # 获得模型
loss = nn.MSELoss()  # 损失函数
optim = torch.optim.SGD(net.parameters(), lr=0.05)  # 随机梯度下降优化器

def datagen(bsz, bnum):
    for _ in range(bsz):
        data = torch.randint(1, 10, (bnum, 10))
        data[:, 0] = 1
        yield Batch(data, data, 2)

net.train()
for epoch in range(40):  # 训练20轮
    running_loss = 0.0
    for d in datagen(10, 8):
        x = d.src
        y = d.trg
        tgt = d.trg_float
        outputs = net(x, y, tgt_mask = d.trg_mask)
        result_loss = loss(outputs, tgt)  # 计算loss
        optim.zero_grad()  # 梯度清零
        result_loss.backward()  # loss反向传播
        optim.step()  # 参数更新
        running_loss = running_loss + result_loss  # 此处演示loss的下降， 代表训练的逐步优化
    print(running_loss)

torch.save(net.state_dict(), "./models_pth/test1.pth")