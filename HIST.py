"""
2024.1.30 16:30
HIST

参考微软HIST图神经网络模型，对原模型中的预定义概念层和隐藏概念层进行合并，用一个隐藏概念层代替，以让机器自主学习股票间的关联关系，
从而节省了预定义概念层的人工标注成本。

模型先通过一个双层的rnn网络提取特征，再通过一个隐藏概念层，rnn的特征结果与隐藏概念层结果的残差传入个体信息层，
最后将个体信息层的结果与隐藏概念层的结果相加并激活，得到最终预测。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def cal_cos_similarity(x_in, y_in):
    x_norm = torch.sqrt(torch.sum(x_in ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y_in ** 2, dim=1))
    cos = torch.mm(x_in, y_in.t()) / torch.mm(x_norm.view(-1, 1), y_norm.view(1, -1) + 1e-6)
    return cos


class HIST(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU", **kwargs):
        super().__init__()

        self.d_feat = d_feat
        self.bidirectional = kwargs.get("双向")
        self.bi_ind = 1 + int(self.bidirectional)
        self.hidden_size = hidden_size * self.bi_ind

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=self.bidirectional,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=self.bidirectional,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc_hidden_out = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        init.xavier_uniform_(self.fc_hidden_out.weight)
        self.fc_hidden_out_fore = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        init.xavier_uniform_(self.fc_hidden_out_fore.weight)
        self.fc_hidden_out_back = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        init.xavier_uniform_(self.fc_hidden_out_back.weight)
        self.fc_individual = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        init.xavier_uniform_(self.fc_individual.weight)

        self.final_fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_input):
        input_hidden, _ = self.rnn(x_input)
        input_hidden = input_hidden[:, -1, :]
        cos_mat = cal_cos_similarity(input_hidden, input_hidden)

        dim = cos_mat.shape[0]
        diag = torch.diag(cos_mat)
        cos_mat_without_diag = cos_mat * (torch.ones(dim, dim) - torch.eye(dim))
        row = torch.linspace(0, dim - 1, dim).long()

        column = cos_mat_without_diag.max(1)[1].long()
        value = cos_mat_without_diag.max(1)[0]

        cos_mat_without_diag[row, column] = 100
        cos_mat_without_diag[cos_mat_without_diag != 100] = 0
        cos_mat_without_diag[row, column] = value

        cos_mat1 = cos_mat_without_diag + torch.diag_embed((cos_mat_without_diag.sum(0) != 0).float() * diag)
        cos_mat2 = torch.mm(cos_mat1.t(), input_hidden)

        cos_mat_new = cal_cos_similarity(cos_mat2, input_hidden)
        cos_mat_new = self.softmax(cos_mat_new)
        output = torch.mm(cos_mat_new, input_hidden)
        output = self.fc_hidden_out(output)

        output_fore = self.fc_hidden_out_fore(output)
        output_fore = F.leaky_relu(output_fore)
        output_back = self.fc_hidden_out_back(output)

        individual_in = input_hidden - output_back
        individual_out = F.leaky_relu(self.fc_individual(individual_in))

        final_out = output_fore + individual_out
        pred = self.final_fc(final_out)

        return pred[...]


# test
if __name__ == "__main__":
    x = torch.randn(2000, 60, 200)
    model = HIST(d_feat=200, hidden_size=64, num_layers=3, dropout=0.5, base_model="GRU", 双向=True)
    y = model(x)
    print(y.shape)
