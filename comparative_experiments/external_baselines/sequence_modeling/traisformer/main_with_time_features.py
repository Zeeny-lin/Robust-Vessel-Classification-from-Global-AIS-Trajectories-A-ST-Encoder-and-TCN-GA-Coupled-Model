#!/usr/bin/env python
# coding: utf-8
"""Pytorch implementation of TrAISformer with Time Features---A generative transformer for
AIS trajectory prediction with time interval and time window features

Enhanced version of the original TrAISformer to support additional time-based features.
"""

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

# 导入新的模块
from models_with_time_features import TrAISformerWithTimeFeatures
from config_time_features import ConfigWithTimeFeatures
from datasets_time_features import AISDatasetWithTimeFeatures, AISDataset_grad_with_time
import utils

# 使用新的配置
cf = ConfigWithTimeFeatures()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
            
        # 数据预处理 - 适应新的数据格式
        for V in l_pred_errors:
            try:
                # 对于带时间特征的数据，检查基本位置特征
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]  # SOG > threshold
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
            
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset with time features...")
        
        # 使用带时间特征的数据集
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = AISDataset_grad_with_time(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = AISDatasetWithTimeFeatures(Data[phase],
                                                           max_seqlen=cf.max_seqlen + 1,
                                                           device=cf.device)
        
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = TrAISformerWithTimeFeatures(cf, partition_model=None)

    ## Trainer - 需要修改trainers以支持时间特征
    # ===============================
    # 注意：原始的trainers.py可能需要修改以支持6维特征输入
    # 这里先使用简化版本的训练逻辑
    
    ## Training
    # ===============================
    if cf.retrain:
        print("Training model with time features...")
        model = model.to(device)
        optimizer = model.configure_optimizers(cf)
        
        best_loss = float('inf')
        
        for epoch in range(cf.max_epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            print(f"Epoch {epoch + 1}/{cf.max_epochs}")
            pbar = tqdm(aisdls["train"])
            
            for batch_idx, (seqs, masks, seqlens, mmsis, time_starts) in enumerate(pbar):
                seqs = seqs.to(device)
                masks = masks[:, :-1].to(device)  # 移除最后一个mask
                
                # 前向传播
                logits, loss = model(seqs, masks=masks, with_targets=True)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cf.grad_norm_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                # 更新进度条
                pbar.set_description(f"Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            
            # 验证
            model.eval()
            valid_loss = 0
            valid_batches = 0
            
            with torch.no_grad():
                for seqs, masks, seqlens, mmsis, time_starts in aisdls["valid"]:
                    seqs = seqs.to(device)
                    masks = masks[:, :-1].to(device)
                    
                    logits, loss = model(seqs, masks=masks, with_targets=True)
                    valid_loss += loss.item()
                    valid_batches += 1
            
            avg_valid_loss = valid_loss / valid_batches
            print(f"Validation Loss: {avg_valid_loss:.4f}")
            
            # 保存最佳模型
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                torch.save(model.state_dict(), cf.ckpt_path)
                print(f"Saved best model with validation loss: {best_loss:.4f}")

    ## Evaluation
    # ===============================
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(cf.ckpt_path))
    model.eval()
    
    # 简化的评估 - 计算测试集损失
    test_loss = 0
    test_batches = 0
    
    with torch.no_grad():
        for seqs, masks, seqlens, mmsis, time_starts in tqdm(aisdls["test"], desc="Evaluating"):
            seqs = seqs.to(device)
            masks = masks[:, :-1].to(device)
            
            logits, loss = model(seqs, masks=masks, with_targets=True)
            test_loss += loss.item()
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # 生成样本轨迹用于可视化
    print("Generating sample trajectories...")
    model.eval()
    
    with torch.no_grad():
        # 获取一个测试批次
        test_batch = next(iter(aisdls["test"]))
        seqs, masks, seqlens, mmsis, time_starts = test_batch
        
        # 只使用前几个样本
        n_samples = min(5, seqs.size(0))
        sample_seqs = seqs[:n_samples].to(device)
        sample_masks = masks[:n_samples].to(device)
        
        # 使用初始序列
        init_seqs = sample_seqs[:, :init_seqlen, :]
        
        # 简单的生成：使用模型预测下一个点
        predictions = []
        current_seq = init_seqs
        
        for step in range(20):  # 生成20个步骤
            logits, _ = model(current_seq)
            
            # 获取最后一个时间步的预测
            last_logits = logits[:, -1, :]
            
            # 分割logits并取概率最高的选择
            lat_logits, lon_logits, sog_logits, cog_logits, time_int_logits, time_win_logits = \
                torch.split(last_logits, (cf.lat_size, cf.lon_size, cf.sog_size, cf.cog_size, 
                                        cf.time_interval_size, cf.time_window_size), dim=-1)
            
            # 选择概率最高的索引
            lat_pred = torch.argmax(lat_logits, dim=-1, keepdim=True)
            lon_pred = torch.argmax(lon_logits, dim=-1, keepdim=True)
            sog_pred = torch.argmax(sog_logits, dim=-1, keepdim=True)
            cog_pred = torch.argmax(cog_logits, dim=-1, keepdim=True)
            time_int_pred = torch.argmax(time_int_logits, dim=-1, keepdim=True)
            time_win_pred = torch.argmax(time_win_logits, dim=-1, keepdim=True)
            
            # 转换回[0,1)范围
            next_point = torch.cat([lat_pred, lon_pred, sog_pred, cog_pred, time_int_pred, time_win_pred], dim=-1).float()
            next_point = next_point / torch.tensor([cf.lat_size, cf.lon_size, cf.sog_size, cf.cog_size, 
                                                  cf.time_interval_size, cf.time_window_size]).to(device)
            
            # 添加到当前序列
            current_seq = torch.cat([current_seq, next_point.unsqueeze(1)], dim=1)
    
    print("Training and evaluation with time features completed!")
    print(f"Model saved to: {cf.ckpt_path}")
    print(f"Results directory: {cf.savedir}")
    
    # 保存特征统计信息
    feature_stats = {
        'test_loss': avg_test_loss,
        'model_config': {
            'time_interval_size': cf.time_interval_size,
            'time_window_size': cf.time_window_size,
            'n_time_interval_embd': cf.n_time_interval_embd,
            'n_time_window_embd': cf.n_time_window_embd
        }
    }
    
    stats_file = os.path.join(cf.savedir, "time_features_stats.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(feature_stats, f)
    print(f"Feature statistics saved to: {stats_file}")

    print("Done!") 