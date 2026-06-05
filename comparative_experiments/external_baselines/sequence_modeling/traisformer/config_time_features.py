# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration flags for TrAISformer with Time Features.
"""

import os
import pickle
import torch


class ConfigWithTimeFeatures():
    retrain = True
    tb_log = False
    device = torch.device("cuda:0")
#     device = torch.device("cpu")
    
    max_epochs = 50
    batch_size = 32
    n_samples = 16
    
    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36
    
    dataset_name = "ship_classification"  # 修改为船舶分类数据集

    if dataset_name == "ship_classification": #==============================
   
        # 基本特征尺寸
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72
        
        # 时间特征尺寸
        time_interval_size = 100    # 时间间隔特征的离散化大小（可根据实际数据调整）
        time_window_size = 744      # 时间窗口特征大小（30天*24小时=720，最多31天*24小时=744）
        
        # 基本特征embedding维度
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        
        # 时间特征embedding维度
        n_time_interval_embd = 64   # 时间间隔特征embedding维度
        n_time_window_embd = 128    # 时间窗口特征embedding维度
    
        # 坐标范围（需要根据实际数据调整）
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    
    #===========================================================================
    # Model and sampling flags
    mode = "mlp"  # 使用mlp模式，直接处理实值特征而不是离散化索引
                  # 这样更适合分类任务
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"
    top_k = 10 # int or None 
    r_vicinity = 40 # int
    
    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # Data flags
    #===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
    
    
    # model parameters
    #===================================================
    n_head = 8
    n_layer = 8
    
    # 更新总特征和embedding尺寸
    full_size = lat_size + lon_size + sog_size + cog_size + time_interval_size + time_window_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd + n_time_interval_embd + n_time_window_embd
    
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 4 # for DataLoader
    
    filename = f"{dataset_name}"\
        + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"\
        + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"\
        + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}-{time_interval_size}-{time_window_size}"\
        + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}-{n_time_interval_embd}-{n_time_window_embd}"\
        + f"-head-{n_head}-{n_layer}"\
        + f"-bs-{batch_size}"\
        + f"-lr-{learning_rate}"\
        + f"-seqlen-{init_seqlen}-{max_seqlen}"
        
    savedir = "./results/"+filename+"/"
    
    ckpt_path = os.path.join(savedir,"model.pt")
    
    # 分类任务相关配置
    num_classes = 4  # 船舶类型数量
    gradient_accumulation_steps = 4
    mixed_precision = True
    gradient_checkpointing = True
    patience = 10
    data_root = r'G:\data_wash_enhanced'  # 数据根目录 