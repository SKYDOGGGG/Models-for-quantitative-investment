# Models for quantitative investment

## Attention-based LSTM/GRU models

2024.1.23 17:00  Alstm1

网络结构分为注意力网络和普通网络，普通网络中加入了一层在时间维度上进行卷积的卷积层，一维卷积层在因子维度上沿着时间步（交易日）移动；

---
2024.1.24 15:00  Alstm2

引入Bahdanau Attention机制，模型由卷积网络、后接注意力机制的RNN（GRU）网络组成。

Attention用于计算注意力分数和上下文向量。
它使用三个线性层（w_q, w_k, w_v）来计算查询（query）和键（keys）之间的关系，然后产生注意力分数和上下文向量。
上下文向量是每个股票的每个隐藏层在时间维度上的加权值，权重由注意力分数决定，格式为[batch_size, hid_size]。

整体上，输入数据同时通过：1.卷积神经网络（包含卷积层），然后通过一个全连接层；2. GRU 网络，其输出与最后一个隐藏状态一起被送入注意力模块，再经过 Dropout 层。

最后，卷积网络和 GRU 网络的输出被合并并通过最后一个全连接层，产生最终的输出。

模型传参有一定变化，需要传入时间长度time_period。网络结构参考：

![image](https://github.com/SKYDOGGGG/miyuan/assets/140141758/9da751b8-cb34-4c4b-9bf5-d43a026da48d)

参考F. Karim, S. Majumdar, H. Darabi and S. Chen, LSTM Fully Convolutional Networks for Time Series Classification

---
2024.1.25 14:30  Alstm3

引入多头注意力机制(替换alstm2中的Bahdanau)，模型由卷积网络和引入注意力机制的RNN（GRU或LSTM）组成。

多头注意力：将注意力机制的计算过程分为多个头部，每个头部都有自己的注意力分数和上下文向量，最后将多个头部的上下文向量拼接起来，再通过一个线性层进行变换，得到最终的上下文向量。网络结构与Alstm2类似。

可能优势：可以多个头独立计算，提高并行度，同时可以让每个头部关注不同的特征。

更新：取消time_period参数以统一代码

---
