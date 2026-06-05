#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用GridCell地理编码的船舶分类训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import logging
from datetime import datetime
from tqdm import tqdm

from model_with_gridcell_encoding import TransformerWithGridCellEncoding
from dataset_with_discrete_time import ShipDatasetWithDiscreteTime

class ConfigGridCell:
    """GridCell地理编码配置"""
    
    # 数据路径
    data_root = "/mnt/workspace/轨迹识别船型"
    output_dir = "/mnt/workspace/out-gridcell"
    
    # 数据配置
    num_classes = 4
    max_seqlen = 650
    min_seqlen = 50
    
    # 时间特征配置
    max_time_interval = 15658
    max_time_window = 744
    time_interval_bins = 1000
    time_window_bins = 744
    
    # GridCell地理编码参数
    gridcell_spa_embed_dim = 64      # GridCell输出维度
    gridcell_frequency_num = 16      # 频率数量
    gridcell_max_radius = 10000      # 最大半径(米)
    gridcell_min_radius = 10         # 最小半径(米)
    gridcell_ffn_hidden_dim = 256    # FFN隐藏层维度
    
    # 模型配置
    d_model = 512
    nhead = 16
    num_layers = 6
    
    # 训练配置
    batch_size = 32
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    max_epochs = 100
    weight_decay = 0.01
    patience = 15
    gradient_clip_norm = 1.0
    
    # 学习率调度
    use_cosine_scheduler = True
    min_learning_rate_ratio = 0.05
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ship_types = ['集装箱船', '干散货船', '渔船', '油船']
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("🔧 GridCell地理编码模型配置:")
        print(f"  数据路径: {cls.data_root}")
        print(f"  输出路径: {cls.output_dir}")
        print(f"  序列长度: {cls.min_seqlen}-{cls.max_seqlen}")
        print(f"  批大小: {cls.batch_size}")
        print(f"  学习率: {cls.learning_rate}")
        
        print(f"\n🌍 GridCell地理编码:")
        print(f"  输出维度: {cls.gridcell_spa_embed_dim}")
        print(f"  频率数量: {cls.gridcell_frequency_num}")
        print(f"  半径范围: {cls.gridcell_min_radius}-{cls.gridcell_max_radius}米")
        print(f"  FFN隐藏维度: {cls.gridcell_ffn_hidden_dim}")
        
        print(f"\n⏱️  时间特征离散化:")
        print(f"  时间间隔: 最大{cls.max_time_interval}秒 → {cls.time_interval_bins}个bin")
        print(f"  时间窗口: 最大{cls.max_time_window}小时 → {cls.time_window_bins}个bin")
        
        print(f"\n🏗️  模型结构:")
        print(f"  GridCell编码: {cls.gridcell_spa_embed_dim}维")
        print(f"  其他连续特征: 2维 (sog, cog)")
        print(f"  时间嵌入: {cls.time_interval_bins} + {cls.time_window_bins} 词汇表")
        print(f"  模型维度: {cls.d_model}")
        print(f"  注意力头: {cls.nhead}")
        print(f"  层数: {cls.num_layers}")

def main():
    """主训练函数"""
    config = ConfigGridCell()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(config)
    
    logger.info("🚀 开始训练 - GridCell地理编码模型")
    logger.info(f"设备: {config.device}")
    config.print_config()
    
    # 数据集
    logger.info("加载数据集...")
    try:
        full_dataset = ShipDatasetWithDiscreteTime(
            data_root=config.data_root,
            max_seqlen=config.max_seqlen,
            min_seqlen=config.min_seqlen
        )
        logger.info(f"总样本数: {len(full_dataset)}")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return
    
    # 数据分割
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.data[i]['label'] for i in indices]
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # 模型
    logger.info("初始化GridCell地理编码模型...")
    model = TransformerWithGridCellEncoding(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        max_seqlen=config.max_seqlen,
        max_time_interval=config.max_time_interval,
        max_time_window=config.max_time_window,
        time_interval_bins=config.time_interval_bins,
        time_window_bins=config.time_window_bins,
        gridcell_spa_embed_dim=config.gridcell_spa_embed_dim,
        gridcell_frequency_num=config.gridcell_frequency_num,
        gridcell_max_radius=config.gridcell_max_radius,
        gridcell_min_radius=config.gridcell_min_radius,
        gridcell_ffn_hidden_dim=config.gridcell_ffn_hidden_dim
    ).to(config.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"嵌入层信息: {model.get_embedding_info()}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * config.min_learning_rate_ratio
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0
    patience_counter = 0
    
    logger.info("开始训练...")
    
    for epoch in range(config.max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.max_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, config.device
        )
        
        # 学习率调度
        if config.use_cosine_scheduler:
            scheduler.step()
        else:
            scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_acc, config, is_best=True)
            logger.info(f"🎉 新的最佳验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config.patience:
            logger.info(f"早停触发，最佳验证准确率: {best_val_acc:.4f}")
            break
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, best_val_acc, config)
    
    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")

def custom_collate_fn(batch):
    """collate函数"""
    sequences = torch.stack([item['sequence'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    shipnos = [item['shipno'] for item in batch]
    
    return {
        'sequence': sequences,
        'mask': masks,
        'label': labels,
        'shipno': shipnos
    }

def setup_logging(config):
    """设置日志"""
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gridcell_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """训练函数"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        sequences = batch['sequence'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        logits = model(sequences, masks)
        loss = criterion(logits, labels)
        
        # 梯度累积
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * config.gradient_accumulation_steps
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        if len(all_labels) > 0:
            current_acc = accuracy_score(all_labels, all_preds)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(sequences, masks)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, best_acc, config, is_best=False):
    """保存检查点"""
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'config': config.__dict__,
        'embedding_info': model.get_embedding_info()
    }
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_gridcell_model.pth')
        torch.save(checkpoint, best_path)

if __name__ == "__main__":
    main()