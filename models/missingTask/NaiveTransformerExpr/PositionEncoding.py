import torch.nn as nn
import torch
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_position=100):
        
#总之，d_hid 是位置编码向量的维度，而 n_position 是位置编码表中的位置数量，这决定了模型能够处理的最大序列长度。
        super(PositionEncoding, self).__init__()
         # 创建一个缓冲区，保存位置编码表。
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

     # 生成正弦波形的位置编码表。
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            # 根据位置和维度生成角度值。
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(int(pos_i)) for pos_i in range(n_position)])# 生成初始位置编码表。
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i# 对表中的偶数索引应用正弦函数。
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1# 对表中的奇数索引应用余弦函数。

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [1,N,d]

    # 前向传播函数，将位置编码添加到输入张量中。
    def forward(self, x):
        # x [B,N,d]
        # print(x.shape ,self.pos_table[:, :x.size(1)].shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


if __name__ == '__main__':
    batch_size, seq_len, n_hid = 16, 10, 128
    x_ = torch.zeros(batch_size, seq_len, n_hid)
    pe = PositionEncoding(d_hid=n_hid, n_position=seq_len)
    y_ = pe(x_)
    print(y_.shape)
