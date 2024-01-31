"""
2024.1.31 10:30
DURNN_1

在DURNN基础上改动，模型先用多重的rnn网络提取时间序列特征，特征传入卷积网络后，将rnn网络的原始输出特征和卷积网络输出合并，激活后输出（激活函数使用LeakyRelu）

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNBase_1(nn.Module):
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


class DURNN_1(nn.Module):
    def __init__(self, d_feat=6, time_period=60, hidden_size=64, dup_num=2, rnn_layers=2, dropout=0.5, 双向=False,
                 **kwargs):
        super().__init__()
        self.input_size = d_feat
        self.hid_size = hidden_size
        self.dup_num = dup_num
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.time_period = time_period
        self.bidirectional = 1 + int(双向)
        self.rnnbase = RNNBase_1(d_feat=self.input_size, hidden_size=self.hid_size, dup_num=dup_num,
                                 rnn_layers=rnn_layers, dropout=dropout, bidirectional=双向)
        self.cnnbase = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_size * self.bidirectional, out_channels=self.hid_size * self.bidirectional,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.rnn_out_fc = nn.Linear(in_features=self.hid_size * self.bidirectional,
                                    out_features=self.hid_size * self.bidirectional)
        self.cnn_out_fc = nn.Linear(in_features=self.time_period, out_features=1)
        self.out_fc = nn.Linear(in_features=self.hid_size * self.bidirectional * 2, out_features=1)

    def forward(self, input_x):
        rnn_out = self.rnnbase(input_x)
        out = self.rnn_out_fc(rnn_out[:, -1, :])
        cnn_out = self.cnnbase(rnn_out.permute(0, 2, 1))
        out_ = self.cnn_out_fc(cnn_out).squeeze(-1)

        final_out = self.out_fc(torch.cat([out, out_], dim=-1))

        return final_out[...]


# test
if __name__ == '__main__':
    x = torch.randn(1000, 60, 100)
    model = DURNN_1(d_feat=100, time_period=60, hidden_size=64, dup_num=2, rnn_layers=2, dropout=0.3, 双向=False)
    y = model(x)
    print(y.shape)
