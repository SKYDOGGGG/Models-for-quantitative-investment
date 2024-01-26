# Models for quantitative investment

Attention-based LSTM/GRU models

"""
2024.1.23 17:00
Alstm1
网络结构分为注意力网络和普通网络，普通网络中加入了一层在时间维度上进行卷积的卷积层，一维卷积层在因子维度上沿着时间步（交易日）移动；
注意力网络不变
"""

alstm2：使用BahdananuAttention注意力机制的LSTM-FCN，结构如下
![image](https://github.com/SKYDOGGGG/miyuan/assets/140141758/9da751b8-cb34-4c4b-9bf5-d43a026da48d)

参考F. Karim, S. Majumdar, H. Darabi and S. Chen, LSTM Fully Convolutional Networks for Time Series Classification

alstm3：使用多头注意力机制的LSTM-FCN，注意力网络有所调整，整体网络架构与上图架构相近
