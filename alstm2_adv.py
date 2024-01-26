"""
2024.1.26 17:00
Alstm2_adv

在Alstm2的基础上进行改动。

在网络结构层面上引入对抗训练，实现方式是将原本的rnn-attention网络路径复制，在复制的路径最后加入梯度反转层
（该层不改变输出，但是在梯度反传时会给梯度乘一个负权重，改变其符号）然后将两个路径的输出拼接起来，送入全连接层进行最终的输出。

梯度反转层中乘的权重是一个随着训练次数增加而增加的值，该值在0到0.5之间，初始值为0，每次训练后增加0.1，最大值为0.5，
这是为了让模型在对抗性强度不断增加的情况下保持性能，增强泛化能力。

梯度反转层适用于存在较大过拟合风险的场景，它可以减少对特定训练集中数据的依赖，提高模型泛化性能。
相比于在训练样本中加入噪声，这种方式更加简单，且不会改变训练样本的分布。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.hid_size = input_size
        self.w_q = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        self.w_k = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        self.w_v = nn.Linear(in_features=input_size, out_features=1, bias=False)

    def forward(self, query, keys):
        query = query.unsqueeze(0)  # [1, batch_size, hid_size]
        query = query.permute(1, 0, 2)  # [batch_size, 1, hid_size]
        energy = self.w_v(torch.tanh(self.w_q(query) + self.w_k(keys)))  # [seq_len, batch_size, hid_size]
        attention = energy.squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        attention_score = attention_weights.unsqueeze(2)  # [seq_len, batch_size, 1]
        context_vector = torch.sum(keys * attention_score, dim=1)  # [batch_size, hid_size]
        return context_vector, attention_weights


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, time_period=60, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.6,
                 rnn_type="GRU"):
        super().__init__()
        self.hid_size1 = hidden_size1
        self.hid_size2 = hidden_size2
        self.time_period = time_period
        self.input_size = d_feat
        self.dropout = dropout
        self.attention = Attention(input_size=self.hid_size2)
        self.attention_adv = Attention(input_size=self.hid_size2)
        self.rnn_type = rnn_type.lower()
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hid_size1, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_size1, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.hid_size1, out_channels=self.hid_size2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_size2, momentum=0.99, eps=1e-3),
            nn.ReLU(),
        )

        self.adv_layer = GradRevLayer(lamda=0.1, max_alpha=0.5)

        self.fc1 = nn.Linear(in_features=self.time_period, out_features=1)
        self.fc_out = nn.Linear(in_features=self.hid_size2 * 2, out_features=1)

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.rnn_adv = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.lstm_drop = nn.Dropout(p=self.dropout)
        self.lstm_drop_adv = nn.Dropout(p=self.dropout)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.zeros_(param)

    def forward(self, inputs):
        # inputs: [batch_size, input_size, input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = self.conv_net(inputs)  # [batch_size, hid_size2, input_day]
        conv_out = self.fc1(conv_out)

        inputs_lstm = inputs.clone().permute(0, 2, 1)  # [batch_size, input_size, input_day]

        if self.rnn_type == "lstm":
            rnn_out, (h, c) = self.rnn(inputs_lstm)
            rnn_out_adv, (h_adv, c_adv) = self.rnn_adv(inputs_lstm)
        elif self.rnn_type == "gru":
            rnn_out, h = self.rnn(inputs_lstm)
            rnn_out_adv, h_adv = self.rnn_adv(inputs_lstm)
        else:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type)

        context, attention_score = self.attention(h[-1], rnn_out)
        out_lstm = self.lstm_drop(context)

        context_adv, attention_score_adv = self.attention_adv(h_adv[-1], rnn_out_adv)
        out_lstm_adv = self.lstm_drop_adv(context_adv)

        out = torch.cat((conv_out, out_lstm.unsqueeze(2)), dim=2)
        out = out.view(len(out), -1)
        out = self.fc_out(out)

        return out[...]


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.neg() * alpha
        return grad_input, None


class GradRevLayer(nn.Module):
    def __init__(self, lamda=0.1, max_alpha=0.5):
        super().__init__()
        self.lamda = lamda
        self.alpha = torch.tensor(0.0, requires_grad=False)
        self.max_alpha = torch.tensor(float(max_alpha), requires_grad=False)
        self.lamda_mul = 0
        self.forward_counter = 0
        self.update_frequency = 30

    def set_alpha(self):
        self.lamda_mul += 1
        self.alpha = min(self.max_alpha, torch.tensor(2.0 / (1.0 + np.exp(-self.lamda * self.lamda_mul)) - 1.0,
                                                      requires_grad=False))

    def forward(self, inputs):
        if self.forward_counter % self.update_frequency == 0:
            self.set_alpha()
        self.forward_counter += 1

        return GradRev.apply(inputs, self.alpha)


# test
if __name__ == "__main__":
    x = torch.randn(20000, 60, 200)
    model = ALSTMModel(d_feat=200, time_period=60, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.6,
                       rnn_type="gru")
    y = model(x)
    print(y.shape)
