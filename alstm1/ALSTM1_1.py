"""
2024.1.29 16:30
ALSTM1_1

在ALSTM1的基础上加入双向rnn，以提高模型性能。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ALSTM1_1(nn.Module):
    def __init__(self, d_feat=130, hidden_size1=256, hidden_size2=128, num_layers=3, dropout=0.9, rnn_type="GRU",
                 双向=False):
        super().__init__()
        self.hid_size1 = hidden_size1
        self.hid_size2 = hidden_size2
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self.bidirectional = 双向
        self.bi_ind = 1 + int(双向)
        self._build_model()
        self.apply(self.init_weights)

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=self.hid_size1, kernel_size=3, padding=1)
        self.fc_in = nn.Linear(in_features=self.hid_size1, out_features=self.hid_size2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.rnn = klass(
            input_size=self.hid_size2,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size2 * 2 * self.bi_ind, out_features=1)
        self.att_fc = nn.Linear(in_features=self.hid_size2 * self.bi_ind, out_features=self.hid_size2 * self.bi_ind)

    def init_weights(self, m):
        import torch.nn.init as init
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

    def attention_net(self, rnn_out):
        attention_weights = F.softmax(self.att_fc(rnn_out), dim=1)
        weighted_hidden = torch.mul(attention_weights, rnn_out)
        return weighted_hidden.squeeze(1)

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = F.leaky_relu(self.conv1(inputs))
        conv_out = conv_out.permute(0, 2, 1)

        out = F.leaky_relu(self.fc_in(conv_out))
        rnn_out, _ = self.rnn(out)
        attention_score = self.attention_net(rnn_out)
        out_att = self.global_pool(attention_score.permute(0, 2, 1)).squeeze(2)

        out = torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        out = self.fc_out(out)

        return out[...]


# test
if __name__ == "__main__":
    x = torch.randn(1000, 30, 200)
    model = ALSTM1_1(d_feat=200, hidden_size1=256, hidden_size2=128, num_layers=2, dropout=0.3, rnn_type="GRU",
                     双向=False)
    y = model(x)
    y = y.detach().numpy()
    print(y.shape)
    print(y)
