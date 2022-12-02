# 进行 copy 任务测试模型效果
import numpy as np
import torch
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode
from Model import make_model

def data_generator(V, batch_size, batch_num):
    """
    V: 随即生成数据的上界 + 1
    batch_size: 一次喂给网络多少个数据样本
    batch_num: 一共喂多少次
    """
    for _ in range(batch_num):
        data = torch.from_numpy(np.random.randint(1, V, size = (batch_size, 10), dtype = "int64"))  # 每一条数据里包含 10 个数
        data[:, 0] = 1  # 将数据的第一列全部设置为 1 ，作为起始标志
        source = Variable(data, requires_grad = False)
        target = Variable(data, requires_grad = False)  # 源数据与目标数据一致并且不随着梯度更新而改变
        
        yield Batch(source, target)

V = 11
model = make_model(V, V, N = 2)
model_optimizer = get_std_opt(model)  # 获得模型的优化器
# 获得标签平滑对象
criterion = LabelSmoothing(size = V, padding_idx = 0, smoothing = 0.0)  
# 获得利用标签平滑的结果得到的损失计算方法
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)  

def run(model, loss, epochs = 10):
    """
    model: 要训练的模型
    loss: 使用的损失计算方法
    epochs: 模型训练的轮次
    """
    for _ in range(epochs):
        model.train()  # 训练模式，参数更新
        run_epoch(data_generator(V, 8, 20), model, loss)  
        
        model.eval()  # 评估模式，参数保留
        run_epoch(data_generator(V, 8, 5), model, loss)
    
    model.eval()
    source = Variable(torch.LongTensor([[1,2,4,3,5,7,6,8,9,10]]))
    source_mask = Variable(torch.ones(1, 1, 10))  # 全 1 的掩码张量，无任何遮掩
    result = greedy_decode(model, source, source_mask, max_len = 10, start_symbol = 1)
    
    # print(result)
    return result

run(model, loss)