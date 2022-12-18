import torch 
from torch import nn 
from torch.nn.modules import Transformer

class Net(nn.Module):
    def __init__(self, vocab, dim) -> None:
        super(Net, self).__init__()
        self.embd = nn.Embedding(vocab, dim)
        self.transformer = Transformer(
            d_model=dim, 
            nhead=2, 
            num_encoder_layers=2, 
            num_decoder_layers=2, 
            dim_feedforward=16, 
            batch_first=True
            )
        self.proj = nn.Linear(dim, 1)

    def forward(self, x, y, tgt_mask):
        x = self.embd(x)
        y = self.embd(y)
        x = self.transformer(x, y, tgt_mask = tgt_mask)
        x = self.proj(x)
        # print(x)
        return torch.squeeze(x)


if __name__ == "__main__":
    data = torch.randint(1, 10, (2, 10))
    data[:, 0] = 0
    x = y = data
    print(x)
    net = Net(10, 6)
    mask = torch.unsqueeze(Transformer.generate_square_subsequent_mask(10), 0)
    for _ in range(3):
        mask = torch.concat((torch.unsqueeze(Transformer.generate_square_subsequent_mask(10), 0), mask))
    # print(mask)
    # print(mask.shape)
    # print(net)
    encode = net.transformer.encoder(net.embd(x))
    print(encode)
    out = net(x, y, tgt_mask = mask)
    print(out)
