"""
2024.1.29 15:30
DURNN

一个多重rnn，可用双向rnn，模型先进行卷积提取特征，后直接传入多重的rnn网络，最后输出

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNBase(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, dup_num=2, rnn_layers=3, dropout=0.3, bidirectional=False):
        super().__init__()
        self.rnn_modules = nn.ModuleList()
        for i in range(dup_num):
            self.rnn_modules.append(nn.GRU(input_size=d_feat, hidden_size=hidden_size, num_layers=rnn_layers,
                                           dropout=dropout, batch_first=True, bidirectional=bidirectional))

    def forward(self, input_x):
        batch_size, seq_len, input_dim = input_x.shape
        outputs = []
        for i in range(len(self.rnn_modules)):
            output, _ = self.rnn_modules[i](input_x)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=-1)
        outputs = torch.mean(outputs, dim=-1)
        return outputs


class DURNN(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, dup_num=2, rnn_layers=2, dropout=0.5, 双向=False):
        super().__init__()
        self.input_size = d_feat
        self.hid_size = hidden_size
        self.dup_num = dup_num
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.bidirectional = 1 + int(双向)
        self.rnnbase = RNNBase(d_feat=self.hid_size, hidden_size=self.hid_size, dup_num=dup_num, rnn_layers=rnn_layers,
                               dropout=dropout, bidirectional=双向)
        self.cnnbase = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hid_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.out_fc = nn.Linear(in_features=self.hid_size * self.bidirectional, out_features=1)

    def forward(self, input):
        cnn_out = self.cnnbase(input.permute(0, 2, 1))
        rnn_out = self.rnnbase(cnn_out.permute(0, 2, 1))
        out = self.out_fc(rnn_out[:, -1, :])

        return out[...]


# test
if __name__ == '__main__':
    x = torch.randn(1000, 60, 100)
    model = DURNN(d_feat=100, hidden_size=64, dup_num=2, rnn_layers=2, dropout=0.3, 双向=True)
    y = model(x)
    print(y.shape)
