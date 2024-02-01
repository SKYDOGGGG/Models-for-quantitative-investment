import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from TRANSFORMER import PositionalEncoding


class TransformerRNN_1(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, rnn_layers=2, rnn_type='GRU', transformer_layers=2, num_heads=2,
                 dropout=0.3, **kwargs):
        super(TransformerRNN_1, self).__init__()
        self.input_size = d_feat
        self.hid_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.feature_layer = nn.Linear(d_feat, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size=hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        self.decoder_layer = nn.Linear(hidden_size, 1)

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.apply(self.init_weights)

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

    def forward(self, input_x):
        # rnn
        rnn_out, _ = self.rnn(input_x)

        # transformer
        rnn_out = rnn_out.transpose(1, 0)

        mask = None  # 所有历史特征都参与计算
        rnn_pos = self.pos_encoder(rnn_out)
        enc_rnn = self.transformer_encoder(rnn_pos, mask)
        dec_rnn = self.decoder_layer(enc_rnn.transpose(1, 0)[:, -1, :])

        return dec_rnn


# test
if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.randn(1000, 60, 100).to(device)
    model = TransformerRNN_1(d_feat=100, hidden_size=64, rnn_layers=2, rnn_type='gru', transformer_layers=2,
                             num_heads=2, dropout=0.3, 双向=False).to(device)
    out = model(x)
    print(out.shape)
