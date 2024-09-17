import os
import sys
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.EMT import EMT

from models.missingTask.transformers_encoder.transformer import TransformerEncoder
from models.missingTask.time import TIMNET
from models.missingTask.blocks import FusionMode


__all__ = ['EMT_DLFR']# 指定模块公开接口

class EMT_DLFR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.aligned = args.need_data_aligned
        # unimodal encoders
        ## text encoder
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

       
         ## projector生成器
        ## gmc_tokens: global multimodal context#简单的投影
        #gmc_tokens_dim = num_modality * args.d_model
        #self.gmc_tokens_projector = Projector(240, 240)
        self.text_projector = Projector(100, 100)
        self.audio_projector = Projector(100, 100)
        self.video_projector = Projector(100,100)

        # self.text_projector_2 = Projector(25, 25)
        # self.audio_projector_2 = Projector(25, 25)
        # self.video_projector_2 = Projector(25,25)

        
        ## predictor预测器
        #这里的 Predictor 通常由两个全连接层组成，中间可能包含一个非线性激活函数。
        #第一个全连接层将输入特征映射到预测层的维度，第二个全连接层则将预测层的输出映射回最终输出特征的维度。
        #self.gmc_tokens_predictor = Predictor(gmc_tokens_dim, args.gmc_tokens_pred_dim, gmc_tokens_dim)
        self.text_predictor = Predictor(100, 50, 100)
        self.audio_predictor = Predictor(100, 50, 100)
        self.video_predictor = Predictor(100, 50, 100)
        
       #
        self.weight_l = nn.Linear(100, 100)
        self.weight_v = nn.Linear(100, 100)
        self.weight_a = nn.Linear(100, 100)
        
        
        

        # low-level feature reconstruction
        #feature_dims': (768, 5, 20),
        self.recon_text = nn.Linear(100, args.feature_dims[0])
        self.recon_audio = nn.Linear(100, args.feature_dims[1])
        self.recon_video = nn.Linear(100 ,args.feature_dims[2])
        

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')
    

        
        self.TIMNET_text = TIMNET(in_channels=768,nb_filters=25)
        self.TIMNET_audio = TIMNET(in_channels=33,nb_filters=25)
        self.TIMNET_video = TIMNET(in_channels=709,nb_filters=25)
        
        self.model_VL = FusionMode(hidden_size=100, num_attention_heads=10, attention_dropout=0.01, intermediate_size=50, bottleneck=50)  #本来是0.1
        
        
        
        self.proj1 = nn.Linear(100, 100)
        self.proj2 = nn.Linear(100, 1)
        # self.proj3 = nn.Linear(100, 1)
        #self.out_layer = nn.Linear(378, 1)
        self.output_dropout = 0.5
        #---注意力参数---可更改 根据不同数据集
      

    def get_network(self, self_type='l', layers=-1):
        self.num_heads = 1
        self.layers = 1
        self.attn_dropout = 0.3
        self.attn_dropout_a = 0.2
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.0
        self.embed_dropout = 0.2
        self.res_dropout = 0.0
        self.output_dropout = 0.5
        self.text_dropout = 0.5
        self.attn_mask = "true"
        
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = 768, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = 33, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = 709, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_text, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_audio, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_video, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing):
        # unimodal encoders
        # print(text.shape)
        # print(audio.shape)
        # print(video.shape)
        # torch.Size([32, 3, 50])
        # torch.Size([32, 375, 5])
        # torch.Size([32, 500, 20])
        
        text = self.text_model(text)#torch.Size([32, 50, 768])
        text_utt, text = text[:,0], text[:, 1:] # (B, 1, D), (B, T, D)# 分离CLS标记和其余文本([32, 1,768])   torch.Size([32, 49, 768])
        text_for_recon = text.detach() # 用于重构的文本特征返回一个新的 Tensortorch.Size([32, 49, 768])
        

        #换位置
        text = text.permute(1, 0, 2)#(50,32,125)
        audio = audio.permute(1, 0, 2)#(375,32,125)
        video = video.permute(1, 0, 2)#(500,32,125)


        #------注意力
        text = self.self_attentions_c_l(text)#torch.Size([50, 32, 125])
        # if type(text_f_att) == tuple:
        #     text_f_att = text_f_att[0]
        # text_f_att = text_f_att[-1] #torch.Size([32, 125])
        
        
        audio = self.self_attentions_c_a(audio)
        # if type(audio_f_att) == tuple:
        #     caudio_f_att = audio_f_att[0]
        # audio_f_att = audio_f_att[-1]#torch.Size([32, 125])课作为全局

        video = self.self_attentions_c_v(video)
        # if type(video_f_att) == tuple:
        #     video_f_att = video_f_att[0]
        # video_f_att = video_f_att[-1] #torch.Size([32, 125])
 
 #换位置
        text = text.permute(1, 0, 2)#(50,32,125)
        audio = audio.permute(1, 0, 2)#(375,32,125)
        video = video.permute(1, 0, 2)#(500,32,125)

        text_TIMNET,text_f = self.TIMNET_text(text)#torch.Size([32, 38, 100])
        audio_TIMNET,audio_f = self.TIMNET_audio(audio)#torch.Size([32, 400, 100])
        video_TIMNET,video_f = self.TIMNET_video(video)#torch.Size([32, 55, 100])




        # 对第二维进行平均，以减少其大小到1
        text_TIMNET_mean = text_TIMNET.mean(dim=1, keepdim=True)#torch.Size([32, 1, 100])
        audio_TIMNET_mean = audio_TIMNET.mean(dim=1, keepdim=True)
        video_TIMNET_mean = video_TIMNET.mean(dim=1, keepdim=True)
        
        text_f_att = text_TIMNET_mean.view(text_TIMNET_mean.size(0),-1)
        audio_f_att = audio_TIMNET_mean.view(text_TIMNET_mean.size(0),-1)
        video_f_att = video_TIMNET_mean.view(text_TIMNET_mean.size(0),-1)

        last_pre = self.model_VL(audio_TIMNET_mean,video_TIMNET_mean, text_TIMNET_mean)
        
        
       
       
        
        ## projector投影器
        
        z_text = self.text_projector(text_f_att)# 文本投影器orch.Size([32, 768])
        z_audio = self.audio_projector(audio_f_att)#torch.Size([32, 16])
        z_video = self.video_projector(video_f_att)#torch.Size([32, 32])

       
        ## predictor维度不变，生成器（2层全连接，隐藏层降一半维度）
        
        p_text = self.text_predictor(z_text)
        p_audio = self.audio_predictor(z_audio)
        p_video = self.video_predictor(z_video)
        #最后的全连接
        last_pre_kd = last_pre.view(last_pre.size(0),-1)

        
        output_fusion = self.proj2(
            F.dropout(F.relu(self.proj1(last_pre), inplace=True), p=self.output_dropout, training=self.training))
        # output_fusion += last_pre
        # output_fusion = self.proj3(output_fusion)
        
        #KD损失的计算
        #以下得到的所有输出 用于返回
# #读于分类之后取log
#         t_log_prob = F.log_softmax(text_TIMNET_mean, 2)
#         a_log_prob = F.log_softmax(audio_TIMNET_mean, 2)
#         v_log_prob = F.log_softmax(video_TIMNET_mean, 2)

#         all_log_prob = F.log_softmax(last_pre_all, 2)
        
#


        # kl_t_log_prob = F.log_softmax(text_TIMNET_mean /1, 2)
        # kl_a_log_prob = F.log_softmax(audio_TIMNET_mean /1, 2)
        # kl_v_log_prob = F.log_softmax(video_TIMNET_mean /1, 2)

        # kl_all_prob = F.softmax(last_pre_all /1, 2)

     
        







        suffix = '_m' if missing else ''#如果某些模态数据缺失（missing 为真），则在键名后会加上后缀 _m
        res = {
            f'pred{suffix}': output_fusion,#如果没有缺失，只用返回这一个预测值。若缺失了，
            
            f'z_text{suffix}': z_text.detach(),#----h
            f'p_text{suffix}': p_text,
            f'z_audio{suffix}': z_audio.detach(),
            f'p_audio{suffix}': p_audio,
            f'z_video{suffix}': z_video.detach(),
            f'p_video{suffix}': p_video,
#----------损失-------------------------------------------
            f'kl_t_log_prob{suffix}': text_f_att,#----h 
            f'kl_a_log_prob{suffix}': audio_f_att,
            f'kl_v_log_prob{suffix}': video_f_att,
            f'kl_all_prob{suffix}': last_pre_kd,
            
        }
        #缺失的话，进行低层特征重构
        # low-level feature reconstruction
        if missing:
            text_recon = self.recon_text(text_TIMNET)#Z_a全连接层 768
            audio_recon = self.recon_audio(audio_TIMNET)#5
            video_recon = self.recon_video(video_TIMNET)#20
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,
                    
                    
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})
       
        return res

    def forward(self, text, audio, video):
       # 将模型输入分为三个部分：文本（包括完整和缺失视图）、音频（包括完整和缺失视图及其长度）、视觉（包括完整和缺失视图及其长度），
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video
        #mosi text     torch.Size([32, 3, 50])
              #audio   torch.Size([32, 375, 5])
              #video    torch.Size([32, 500, 20])

        
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2 # -2 for CLS and SEP.torch.Size([32])

    
        # complete view 对完整和缺失输入的网络处理
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # incomplete view
        res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True)

        return {**res, **res_m}# 将完整视图和缺失视图的结果合并为一个字典并返回。


#------------------------------------------------
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # hidden layer
                                 nn.Linear(pred_dim, output_dim))  # output layer

    def forward(self, x):
        return self.net(x)


