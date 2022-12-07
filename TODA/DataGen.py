import numpy as np 
import torch
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def dist(A, B):
    """计算 A B 矩阵行向量之间的欧式距离"""
    m, n = A.shape
    t, n1 = B.shape
    if n != n1: return
    ones1 = np.ones((n, t))
    ones2 = np.ones((n, m))
    dist = np.sqrt(np.matmul(np.square(A), ones1) + \
                   np.transpose(np.matmul(np.square(B), ones2)) \
                   - 2 * np.matmul(A, np.transpose(B)))
    return dist

def data_gen(detectors, tgt_sig, re_num, batch_size = 8, batch_num = 10, to_batch = True):
    """
    detectors: 接收器坐标
    tgt_sig: 目标发射的信号
    re_num: 接收器接收的信号点数
    batch_size: 一个batch有多少条数据
    batch_num: 多少个batch
    to_batch: 是否返回batch类型
    """
    for _ in range(batch_num):
        xy = np.random.random((batch_size, 2))  # 随机生成 x, y
        z = np.zeros(batch_size)  # z坐标为0
        tgt = np.column_stack((xy, z))  # 三维坐标
        tgt = np.around(tgt, 3)  # 保留三位小数  
        d = dist(tgt, detectors)  # 计算距离
        # 衰减系数, 此处用简单的线性拟合, 具体建模见note
        decay = 1.09282 * (1 - d) + 1 
        start = np.array(d * 100, dtype = "int")  # 计算信号到达时延
        data = None
        for i, s in enumerate(start):
            if i == 0: 
                data = np.concatenate(tuple(tgt_sig[s[j]: s[j] + re_num]\
                    * decay[i][j] for j in range(len(s))), 0)
            else:
                data = np.vstack((data, np.concatenate(tuple((tgt_sig[s[j]: s[j] + re_num]\
                    * decay[i][j] for j in range(len(s)))), 0)))
        
        if data.shape[0] != batch_size: data = np.array([data])
        data[:, 0] = 0  # 起始位
        # 放大 1000 倍至整数区间，方便embd
        tgt = Variable(
            torch.from_numpy(np.array(tgt * 1000, dtype = "int64")), 
            requires_grad = False
            )
        data = Variable(
            torch.from_numpy(np.array(data * 1000, dtype = "int64")), 
            requires_grad = False
            )  
        
        if to_batch: yield Batch(data, tgt)
        else: yield (data, tgt)



if __name__ == "__main__":
    detectors = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    tgt_sig = np.zeros(400)
    for i in range(10):
        tgt_sig[i::20] = 1
    re_num = 100
    res = data_gen(detectors, tgt_sig, re_num, batch_size = 1, batch_num = 1)
    
    # for i, r in enumerate(res):
    #     if i == 3: 
    #         print(r[0], "\n", r[1])
    #         print(r[0].shape, "\n", r[1].shape)
    #     print(r[0].shape, r[1].shape)
    # print(i)

    for d in res:
        # src: 输入数据，此处对应波形
        # trg：目标输出，此处对应坐标
        print(d.src, d.trg)
    print(type(d.src), d.src.shape[-1])