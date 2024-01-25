# miyuan
models

alstm1: 在rnn前面加了一层卷积层，简单的调了些参数，先试试能不能跑 

alstm2：使用BahdananuAttention注意力机制的LSTM-FCN，结构如下
![image](https://github.com/SKYDOGGGG/miyuan/assets/140141758/9da751b8-cb34-4c4b-9bf5-d43a026da48d)

参考F. Karim, S. Majumdar, H. Darabi and S. Chen, LSTM Fully Convolutional Networks for Time Series Classification

alstm3：使用多头注意力机制的LSTM-FCN，注意力网络有所调整，整体网络架构与上图架构相近
