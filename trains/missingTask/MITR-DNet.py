import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class EMT_DLFR():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.criterion = nn.L1Loss()
        self.MSE = MSE()
    # ------------定义一个内部函数用于计算模型中包含fusion的参数数量
    def do_train(self, model, dataloader):
        def count_parameters(model, name='.fusion'):
            answer = 0
            for n, p in model.named_parameters():
                if name in n:
                    answer += p.numel()
            return answer
        # 定义第二个内部函数用于计算除去特定名称的参数外的模型参数数量
        def count_parameters_2(model):
            answer = 0
            for n, p in model.named_parameters():
                if 'predictor' not in n and 'projector' not in n and 'recon' not in n:
                    answer += p.numel()
            return answer

        logger.info(f'The model during inference has {count_parameters_2(model)} parameters.')# 记录模型推理时的参数数量
        logger.info(f'The fusion module (emt) has {count_parameters(model)} parameters.') # 记录fusion模块的参数数量

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())# 获取文本模型的所有参数
        audio_params = list(model.Model.TIMNET_audio.named_parameters())
        video_params = list(model.Model.TIMNET_video.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'TIMNET_audio' not in n and 'TIMNET_video' not in n and 'text_model' not in n]


        optimizer_grouped_parameters = [ # 定义优化器参数组
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)# 初始化Adam优化器



        # criterion
        criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)# 定义余弦相似度损失函数（高级吸引）
        criterion_recon = ReconLoss(self.args.recon_loss)# 定义重构损失函数
        
        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'Loss(pred_m)'] else 'max'# 根据评估关键词确定是寻找最小值还是最大值，如果是'Loss'或'Loss(pred_m)'，则寻找最小值，否则寻找最大值
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            s_t = time.time()

            with tqdm(dataloader['train']) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)# 加载视觉数据
                    audio = batch_data['audio'].to(self.args.device)# 加载音频数据
                    text = batch_data['text'].to(self.args.device)# 加载文本数据
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)# 加载并将缺失的视觉数据转移到指定的设备
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device) #加载并将视觉数据的缺失掩码转移到指定的设备
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)# 加载并将视觉数据的掩码转移到指定的设备
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)

                    if not self.args.need_data_aligned:# 如果不需要数据对齐
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0 #如果需要数据对齐，则将音频和视觉的长度设置为0

                    # -------------------------------forward----------------------------
                    #训练时有完整和缺失两个数据的输入
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))
                    # store results
                    y_pred.append(outputs['pred_m'].cpu())# 将模型对缺失视图的预测结果添加到预测列表中，同时将数据移动到CPU。
                    y_true.append(labels.cpu())
                    # compute loss
                    ## prediction loss
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))# 计算模型对于缺失视图预测的平均绝对误差损失
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))# 计算模型对于完整视图预测的平均绝对误差损失
                    

                    # ## attraction loss (high-level)
                    # loss_attra_gmc_tokens = -(criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                    #                       criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5# 计算全局多模态上下文(gmc)标记之间的吸引损失
                    loss_attra_text = -(criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                          criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5 #tensor(-0.0213
                    loss_attra_audio = -(criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                          criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5#0.0232
                    loss_attra_video = -(criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                          criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5#0.0360
                    loss_attra =  loss_pred + loss_attra_text+ loss_attra_audio+loss_attra_video  #tensor(1.8754
                 
                 


                 ## reconstruction loss (low-level)
                    mask = text[:, 1, 1:] - text_missing_mask[:, 1:] # '1:' for excluding CLS(32,38)
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                   
                    mask = audio_mask - audio_missing_mask#torch.Size([32, 400])和长度是一样
                    loss_recon_audio = criterion_recon(outputs['audio_recon'], audio, mask)

                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'], vision, mask)
                
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video
                    
                    

                    # kd_all
                    kdloss1 = self.criterion(outputs["kl_t_log_prob_m"], outputs["kl_all_prob"])
                    kdloss2 = self.criterion(outputs["kl_a_log_prob_m"], outputs["kl_all_prob"])
                    kdloss3 = self.criterion(outputs["kl_v_log_prob_m"], outputs["kl_all_prob"])
                    kd = kdloss1+ kdloss2 + kdloss3
                    # reconstruction2 loss
                    loss_recon_l = self.MSE(outputs["kl_t_log_prob_m"], outputs["kl_t_log_prob"])
                    loss_recon_v = self.MSE(outputs["kl_a_log_prob_m"], outputs["kl_a_log_prob"])
                    loss_recon_a = self.MSE(outputs["kl_v_log_prob_m"], outputs["kl_v_log_prob"])
                    loss_recon_kd = loss_recon_l + loss_recon_v + loss_recon_a

                    loss_recon_l_m = self.MSE(outputs["kl_t_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_v_m= self.MSE(outputs["kl_a_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_a_m = self.MSE(outputs["kl_v_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_m = loss_recon_l_m + loss_recon_v_m + loss_recon_a_m



                    

                    ## total loss
                    loss = loss_pred_m +  loss_attra +  loss_recon + (kd+loss_recon_kd+loss_recon_m)*0.3

                    loss.backward()

                    if batch_idx % (len(td) // 2) == 0:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Train] Loss {loss:.4f}')
                    train_loss += loss.item()
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            e_t = time.time()
            logger.info(f'One epoch time for training: {e_t - s_t:.3f}s.')

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)# 计算训练结果的指标

            log_infos = [''] * 8
            log_infos[0] = log_infos[-1] = '-' * 100

            # -----------validation验证过程------------
            s_t = time.time()

            val_results = self.do_test(model, dataloader['valid'], criterion_attra, mode="VAL", epochs=epochs)

            e_t = time.time()
            logger.info(f'One epoch time for validation: {e_t - s_t:.3f}s.')

            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                self.best_epoch = best_epoch
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                log_infos[5] = f'==> Note: achieve best [Val] results at epoch {best_epoch}'

            log_infos[1] = f"Seed {self.args.seed} ({self.args.seeds.index(self.args.seed)+1}/{self.args.num_seeds}) " \
                       f"| Epoch {epochs} (early stop={epochs-best_epoch}) | Train Loss {train_loss:.4f} | Val Loss {val_results['Loss']:.4f}"
            log_infos[2] = f"[Train] {dict_to_str(train_results)}"
            log_infos[3] = f"  [Val] {dict_to_str(val_results)}"

            # log information
            for log_info in log_infos:
                if log_info: logger.info(log_info)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                logger.info(f"==> Note: since '{self.args.KeyEval}' does not improve in the past {self.args.early_stop} epochs, early stop the training process!")
                return

    def do_test(self, model, dataloader, criterion_attra=None, criterion_recon=None, mode="VAL", epochs=None):
        if epochs is None: logger.info("=" * 30 + f"Start Test of Seed {self.args.seed}" + "=" * 30)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_loss_pred = 0.0
        # criterion = nn.L1Loss()
        if criterion_attra is None: criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)
        if criterion_recon is None: criterion_recon = ReconLoss(self.args.recon_loss)
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))

                    # compute loss
                    ## task loss (prediction loss of incomplete view)
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    
                    # ## attraction loss (high-level)
                    # loss_attra_gmc_tokens = -(criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                    #                       criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                    loss_attra_text = -(criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                          criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                    loss_attra_audio = -(criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                          criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                    loss_attra_video = -(criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                          criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1))) # prediction loss of complete view

                    loss_attra =  loss_pred + loss_attra_text+ loss_attra_audio+loss_attra_video
                    ## reconstruction loss (low-level)
                    mask = text[:, 1, 1:] - text_missing_mask[:, 1:] # '1:' for excluding CLS
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                   

                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = criterion_recon(outputs['audio_recon'], audio, mask)
                    #kl_loss_audio = kl_loss(kl_lp_2, kl_p_all, mask)

                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'], vision, mask)
                    #kl_loss_video = kl_loss(kl_lp_3, kl_p_all, mask)
                    

                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video
                    
                    # kd_all
                    kdloss1 = self.criterion(outputs["kl_t_log_prob_m"], outputs["kl_all_prob"])
                    kdloss2 = self.criterion(outputs["kl_a_log_prob_m"], outputs["kl_all_prob"])
                    kdloss3 = self.criterion(outputs["kl_v_log_prob_m"], outputs["kl_all_prob"])
                    kd = kdloss1+ kdloss2 + kdloss3
                    # reconstruction2 loss
                    loss_recon_l = self.MSE(outputs["kl_t_log_prob_m"], outputs["kl_t_log_prob"])
                    loss_recon_v = self.MSE(outputs["kl_a_log_prob_m"], outputs["kl_a_log_prob"])
                    loss_recon_a = self.MSE(outputs["kl_v_log_prob_m"], outputs["kl_v_log_prob"])
                    loss_recon_kd = loss_recon_l + loss_recon_v + loss_recon_a

                    loss_recon_l_m = self.MSE(outputs["kl_t_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_v_m= self.MSE(outputs["kl_a_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_a_m = self.MSE(outputs["kl_v_log_prob_m"], outputs["kl_all_prob_m"])
                    loss_recon_m = loss_recon_l_m + loss_recon_v_m + loss_recon_a_m



                    

                    ## total loss
                    loss = loss_pred_m +  loss_attra +  loss_recon + (kd+loss_recon_kd+loss_recon_m)*0.1
                    


                    if batch_idx % (len(td) // 2) == 0:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Val] Loss {loss:.4f}')
                    eval_loss += loss.item()
                    eval_loss_pred += loss_pred_m.item()
                    y_pred.append(outputs['pred_m'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_loss_pred = eval_loss_pred / len(dataloader)
        # logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        # logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        eval_results['Loss(pred_m)'] = eval_loss_pred
        if epochs is None: # for TEST
            logger.info(f"\n [Test] {dict_to_str(eval_results)}")
            logger.info(f"==> Note: achieve this results at epoch {self.best_epoch} (best [Val]) / {getattr(self, 'best_test_epoch', None)} (best [Test])")
        return eval_results

# 定义重构损失函数
class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
       
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss


# class MaskedKLDivLoss(nn.Module): #一个带掩码的 KL 散度损失
#     def __init__(self):
#         super(MaskedKLDivLoss, self).__init__()
#         self.loss = nn.KLDivLoss(reduction='sum') #计算所有元素的 KL 散度损失之和。

#     def forward(self, log_pred, target, mask):
#         mask_ = mask.view(-1, 1)
#         loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
#         return loss
