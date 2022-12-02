import math, copy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torch.nn import ModuleList

class Embeddings(nn.Module):
    """词嵌入类: 
    将输入空间的一个值映射成特征空间中的高维向量,
    希望在高维空间捕捉词汇间的关系
    """
    def __init__(self, d_model, vocab):
        # d_model: 词嵌入的维度
        # vocab: 词表的大小
        super(Embeddings, self).__init__()
        # 定义Embdding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        # x为输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码类"""
    def __init__(self, d_model, dropout, max_len=5000):
        """
        d_model 词嵌入的纬度
        dropout 置零比率
        max_len 句子最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 实例化 dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化一个位置编码矩阵， 大小是 maxlen * d_model
        pe = torch.zeros(max_len, d_model)
        
        # 初始化一个绝对位置矩阵 maxlen * 1
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变化矩阵div_term, 跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 将定义的变化矩阵进行奇数，偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增维
        pe = pe.unsqueeze(0)
        
        # 注册成bufer
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        """# x 代表文本序列的词嵌入表示（即张量表示）"""
        # 将 pe 截取到输入序列的长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)  # 不参与训练更新 
        return self.dropout(x)  # 返回携带了位置编码并丢弃一部分值的结果


def attention(query, key, value, mask = None, dropout = None):
    """注意力的计算"""
    # 此处dropout为一个dropout层对象
    d_k = query.size(-1)  # query的最后一个纬度，即特征向量的纬度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算attention
    # print(scores.size())
    if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)  # 掩码
    p_attn = F.softmax(scores, dim = -1)  # 对最后一格纬度做softmax，即得到attention在特征空间各个分量的分数
    if dropout is not None: p_attn = dropout(p_attn)
    
    # 完成attention的计算
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    """克隆函数"""
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, head, embdding_dim, dropout = 0.1):
        """
        head: 多头注意力的数量
        embdding_dim: 输入的特征纬度
        dropout: 置零的比率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embdding_dim % head == 0  # 确认特征纬度能被head整除
        
        self.head = head
        self.embdding_dim = embdding_dim
        self.d_k = embdding_dim // head  # 每个head分别处理的特征纬度
        # 生成结构中的线性层，共4个
        self.linears = clones(nn.Linear(embdding_dim, embdding_dim), 4)  # 线性层输入输出都是特征的维数
        self.attn = None  # 初始化注意力张量
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, query, key, value, mask = None):
        """query, key, value 是注意力的三个输入张量,mask代表掩码张量"""
        if mask is not None: mask = mask.unsqueeze(1)  # 升维，代表多头中的第n个头
        batch_size = query.size(0)  # 获得batchsize，即channel
        
        """
        未切割之前每个纬度的含义:0 batchsize 即一次性喂了多少个数据； 
        1 length 即该数据(句子的长度); 2 dim 句子中每个元素的特征维度
        model(x). 后面的处理：
        1. 将输出张量的最后一个纬度进行切割,即划分输入给为多个head的特征,此时 2 nth 代表第n个头 3 dim 代表第n个头处理的特征维度
        2. 进行transpose的操作 将 1 2 转置
        """
        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)   
         for model, x in zip(self.linears, (query, key, value))]  # 分别对输入的Q K V进行全连接层的处理
        
        # 将每个头的输出传入到注意力层
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)
        
        # 得到每个头的计算结果是4维张量，需要进行形状的转换
        # 前面已经将1，2两个维度进行过转置，在这里要重新转置回来
        # 经历了transpose方法后，必须要使用contiguous方法，不然无法使用view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embdding_dim)
        
        # 最后对输出进行全连接
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """前馈全连接层"""
    def __init__(self, d_model, d_ff, dropout = 0.1):
        """
        d_model: 词嵌入维度，特征维度
        d_ff: 第一个线性层的输出，第二个线性层的输入
        dropout: 置零比率
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x):
        """
        x 代表上一层的输出
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    """归一化层"""
    def __init__(self, features, eps = 1e-6):
        """
        features: 词嵌入维度
        eps: 一个足够小的正数，用来在规范化计算公式的分母中防止除零操作
        """
        super(LayerNorm, self).__init__()
        
        # 初始化两个参数张量a2，b2 用于对结果做规范化计算
        # 将其用nn.Parameter进行封装，代表他们是模型中的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)  # 在最后一个维度即特征维求均值，保持维度
        std = x.std(-1, keepdim = True)  # 同上
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SubLayerConnection(nn.Module):
    """子层连接结构"""
    def __init__(self, size, dropout = 0.1):
        """size: 词嵌入维度 dropout: 置零比率"""
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p = dropout)
        self.size = size
        
    def forward(self, x, sublayer):
        """x: 上一层的输出 sublayer:该子层中的子层函数,如attention等"""
        # 将上一层输出经过归一化送入子层函数，再dropout以及残差连接
        return x + self.dropout(sublayer(self.norm(x)))  


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        size: 词嵌入维度
        self_attn: 多头注意力子层的实例化对象
        feed_forward: 前馈全连接层的实例化对象
        dropout: 置零比率
        """
        super(EncoderLayer, self).__init__()
        
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)  # 复制两个子层结构
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 第一个自注意力层
        return self.sublayer[1](x, self.feed_forward)  # 第二个前馈全连接层


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, layer, N):
        """layer: 代表解码器层 N:代表解码器中有几个layer"""
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 复制 N 个编码器层
        self.norm = LayerNorm(layer.size)  # 初始化一个规范化层，作用在编码器的最后面
        
    def forward(self, x, mask):
        """x: 上一层输出张量， mask:掩码张量"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        size: 词嵌入维度
        self_attn: 多头自注意力机制对象
        src_attn: 常规注意力机制对象
        feed_forward: 前馈全连接层
        dropout: 置零比率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)  # 复制 3 个子层连接对象
    
    def forward(self, x, memory, source_mask, target_mask):
        """
        x: 上一层的输出
        memory:编码器得到的句子的语义
        source_mask: 源数据的掩码张量 -> 为了遮盖住对结果信息无用的数据
        target_mask: 解码时遮盖住未来的信息，不产生因果问题
        """
        m = memory
        # 对输入进行自注意力操作，同时用掩码遮盖
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))  
        # 对输入进行常规注意力操作，遮盖住不重要的区域
        # 形象理解即此处使用编码层获得的语义信息对目标的文本进行语义的提取处理
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 全连接层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, layer, N):
        """layer: 解码器层的对象； N:堆叠层数"""
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # 即特征维度
    
    def forward(self, x, memory, source_mask, target_mask):
        """
        x: 上一层输出
        memory: 编码器的输出，即语义提取张量
        source_mask: 源数据的掩码张量
        target_mask: 目标数据的掩码张量
        """
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)  # 输出归一化的 x


# 将线性层和softmax计算层一起实现，因为二者的共同目标是生成最后的结构
# 因此把类的名字叫做Generator， 生成器类
class Generator(nn.Module):
    """输出层"""
    def __init__(self, d_model, vocab_size):
        """d_model: 词嵌入维度:vocab_size: 词表大小"""
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)  # 映射到指定维度
        
    def forward(self, x):
        """x: 上一层的输出张量"""
        # return F.softmax(self.project(x), dim = -1)  # 在最后一个维度进行映射操作即特征映射
        return F.log_softmax(self.project(x), dim = -1)  # 另一种softmax


class EncoderDecoder(nn.Module):
    """模型构建"""
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        encoder: 编码器对象
        decoder: 解码器对象
        source_embed: 源数据的嵌入函数
        target_embed:目标数据的嵌入函数
        generator: 输出部分类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    
    def forward(self, source, target, source_mask, target_mask):
        """
        source: 代表源数据
        target: 代表目标数据
        source_mask: 代表源数据的掩码张量
        target_mask: 代表目标数据的掩码张量
        """
        return self.decode(
            self.encode(source, source_mask),
            source_mask,
            target,
            target_mask
        )
    
    def encode(self, source, source_mask):
        # 对源输入进行词嵌入并带着 mask 进行 encode
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        # 对目标的输出进行词嵌入，并且同编码器的输出一同进行解码
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N = 6, d_model = 512, d_ff = 1024, head = 8, dropout = 0.2):
    """
    构建模型
    source_vocab: 代表源数据的词汇总数
    target_vocab: 代表目标数据的词汇总数
    N: 代表编码器和解码器堆叠的层数
    d_model: 代表词嵌入的维度
    d_ff: 代表前馈全连接层中变换矩阵的维度
    head: 多头注意力机制中的头数
    dropout: 置零的比率
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)  # 实例化多头注意力的对象
    ff = PositionwiseFeedForward(d_model, d_ff)  # 实例化全连接层对象
    pe = PositionalEncoding(d_model, dropout)  # 实例化位置编码器
    
    # 实例化模型 model，利用EncoderDecoder类
    # 编码器的结构中有 2 个子层，attention层和全连接层
    # 解码器中有 3 个子层，两个attention和一个全连接层
    # 都各自堆叠 N 次
    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                d_model,  # 词嵌入维度
                c(attn),  # 自注意力层  
                c(ff),   # 全连接层
                dropout
            ), N),  # 堆叠 N 层
        Decoder(
            DecoderLayer(
                d_model,
                c(attn),  
                c(attn),  # 这里的两个注意力对象不同，功能也不同
                c(ff),
                dropout
            ), N),  # 堆叠 N 层
        nn.Sequential(Embeddings(d_model, source_vocab), c(pe)),  # 输入文本的嵌入，加入位置编码
        nn.Sequential(Embeddings(d_model, target_vocab), c(pe)),  # 对目标文本进行词嵌入
        Generator(d_model, target_vocab)  # 由特征向量映射到目标词汇表
    )
    
    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)  # 进行均匀初始化
    
    return model


