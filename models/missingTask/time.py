import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout_rate=0.01):#原来是0.01
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1d(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]  # Adjust if necessary to correctly handle the causal padding
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

# CausalConv1d remains the same as your corrected version.

# TemporalAwareBlock with added kernel_size parameter and removed dynamic Conv1d creation.
class TemporalAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, dropout_rate=0.01):#原来是0.01
        super(TemporalAwareBlock, self).__init__()
        
        self.tanh_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation_rate, dropout_rate)
        self.sigmoid_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation_rate, dropout_rate)
        self.skip_conv = nn.Conv1d(out_channels,out_channels, kernel_size=1)
    def forward(self, input_):
        residual = input_#([32, 125, 50]) 
        tanh_out = self.tanh_conv(input_) #([32, 125, 50]) 
        sigmoid_out = self.sigmoid_conv(input_)#([32, 125, 50]) 
        merged = torch.mul(torch.tanh(tanh_out), torch.sigmoid(sigmoid_out))#([32, 125, 50]) 
        skip_out = self.skip_conv(merged)#([32, 125, 50]) 
        out = skip_out + input_
        return out

class TIMNET(nn.Module):
    def __init__(self, in_channels,nb_filters, kernel_size=2, nb_stacks=1, dilations=None, activation=nn.ReLU(), dropout_rate=0.01):#原来是0.1
        super(TIMNET, self).__init__()
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations if dilations is not None else [2 ** i for i in range(4)]
        #self.dilations =[1,3,5,7]
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.forward_conv = CausalConv1d(in_channels, nb_filters, kernel_size=1,dilation=1)
        self.backward_conv =CausalConv1d(in_channels, nb_filters, kernel_size=1,dilation=1)
        self.temporal_blocks = nn.ModuleList()

        for s in range(nb_stacks):
            for i in self.dilations:
                self.temporal_blocks.append(TemporalAwareBlock(nb_filters,nb_filters, kernel_size, i, dropout_rate))

        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        forward = inputs
        backward = torch.flip(inputs, dims=[1])
        
        forward_convd = self.forward_conv(forward)
        forward_convd_f = forward_convd.permute(0, 2, 1)
        backward_convd = self.backward_conv(backward)
        
        final_skip_connection = []
        
        skip_out_forward = forward_convd

        skip_out_backward = backward_convd
        
        for temporal_block in self.temporal_blocks:
            skip_out_forward = temporal_block(skip_out_forward)
            skip_out_backward = temporal_block(skip_out_backward)
            
            temp_skip = skip_out_forward + skip_out_backward#torch.Size([32, 125, 50])
            
            # temp_skip = torch.mean(temp_skip, dim=2, keepdim=True)
            
            # temp_skip = torch.unsqueeze(temp_skip, dim=2)
            # print(temp_skip.shape)
            # exit()
            final_skip_connection.append(temp_skip)
        
        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection[1:], 1):
            output_2 = torch.cat([output_2, item], dim=1)
        
        x = output_2.permute(0, 2, 1)  #torch.Size([32, 50, 500])
        
        return x,forward_convd_f

# # Example of how to run TIMNET
# # Define model
#model = TIMNET(in_channels=768,nb_filters=125)  # Adjust in_channels according to your input data

# # Create some random input data
# # Assume the input data shape is (batch_size, channels, length), e.g., (1, 1, 100)
#input_data = torch.randn(32,50,768)

# # Run the model
#output = model(input_data)

# print(output.shape)  # This wi

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class WaveNetBlock(nn.Module):
#     def __init__(self, n_atrous_filters, atrous_filter_size, atrous_rate):
#         super(WaveNetBlock, self).__init__()
        
#         self.tanh_conv = nn.Conv1d(n_atrous_filters, n_atrous_filters, kernel_size=atrous_filter_size, dilation=atrous_rate, padding=(atrous_filter_size - 1) * atrous_rate)
#         self.sigmoid_conv = nn.Conv1d(n_atrous_filters, n_atrous_filters, kernel_size=atrous_filter_size, dilation=atrous_rate, padding=(atrous_filter_size - 1) * atrous_rate)
#         self.skip_conv = nn.Conv1d(n_atrous_filters, 1, kernel_size=1)

#     def forward(self, input_):
#         residual = input_
#         tanh_out = self.tanh_conv(input_)
#         print(tanh_out.shape)
#         sigmoid_out = self.sigmoid_conv(input_)
#         print(sigmoid_out.shape)
#         merged = torch.mul(torch.tanh(tanh_out), torch.sigmoid(sigmoid_out))
#         print(merged.shape)
#         skip_out = self.skip_conv(merged)
#         print(skip_out.shape)
#         exit()
#         out = skip_out + input_
#         return out, skip_out


# # 创建一个WaveNetBlock实例
# block = WaveNetBlock(n_atrous_filters=64, atrous_filter_size=2, atrous_rate=2)

# # 生成随机输入数据，假设batch大小为32，时间步数为100，特征维度为64
# input_data = torch.randn(32, 64, 100)

# # 进行前向传播
# output, skip_output = block(input_data)

# # 打印输出的形状
# print("输出的形状:", output.shape)
# print("跳跃连接输出的形状:", skip_output.shape)