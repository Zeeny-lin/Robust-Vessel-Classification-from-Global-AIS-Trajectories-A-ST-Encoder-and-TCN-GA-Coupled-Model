#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用GridCell地理编码模型进行船舶类型预测
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from model_with_gridcell_encoding import TransformerWithGridCellEncoding
from dataset_with_discrete_time import ShipDatasetWithDiscreteTime

class ConfigGridCellPredict:
    """GridCell预测配置"""
    
    # 模型路径
    model_path = "/mnt/workspace/out-gridcell/checkpoints/best_gridcell_model.pth"
    
    # 测试数据路径
    test_data_dir = "/mnt/workspace/轨迹识别船型/test"
    
    # 输出文件
    output_file = "gridcell_predictions.csv"
    
    # 数据配置（需与训练时一致）
    num_classes = 4
    max_seqlen = 650
    min_seqlen = 50
    
    # 时间特征配置
    max_time_interval = 15658
    max_time_window = 744
    time_interval_bins = 1000
    time_window_bins = 744
    
    # GridCell地理编码参数
    gridcell_spa_embed_dim = 256
    gridcell_frequency_num = 32
    gridcell_max_radius = 10000
    gridcell_min_radius = 10
    gridcell_ffn_hidden_dim = 256
    
    # 模型配置
    d_model = 512
    nhead = 16
    num_layers = 6
    
    # 预测配置
    batch_size = 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ship_types = ['集装箱船', '干散货船', '渔船', '油船']

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"📥 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从检查点获取配置
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            print(f"✅ 从检查点加载配置")
        else:
            print("⚠️ 使用默认配置")
            config_dict = ConfigGridCellPredict().__dict__
        
        # 创建模型
        model = TransformerWithGridCellEncoding(
            d_model=config_dict.get('d_model', 512),
            nhead=config_dict.get('nhead', 16),
            num_layers=config_dict.get('num_layers', 6),
            num_classes=config_dict.get('num_classes', 4),
            max_seqlen=config_dict.get('max_seqlen', 650),
            max_time_interval=config_dict.get('max_time_interval', 15658),
            max_time_window=config_dict.get('max_time_window', 744),
            time_interval_bins=config_dict.get('time_interval_bins', 1000),
            time_window_bins=config_dict.get('time_window_bins', 744),
            gridcell_spa_embed_dim=config_dict.get('gridcell_spa_embed_dim', 256),
            gridcell_frequency_num=config_dict.get('gridcell_frequency_num', 32),
            gridcell_max_radius=config_dict.get('gridcell_max_radius', 10000),
            gridcell_min_radius=config_dict.get('gridcell_min_radius', 10),
            gridcell_ffn_hidden_dim=config_dict.get('gridcell_ffn_hidden_dim', 256)
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功加载模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
            print(f"✅ 最佳验证准确率: {checkpoint.get('best_accuracy', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 成功加载模型权重")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def custom_collate_fn(batch):
    """collate函数 - 与训练时保持一致"""
    sequences = torch.stack([item['sequence'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    shipnos = [item['shipno'] for item in batch]
    
    return {
        'sequence': sequences,
        'mask': masks,
        'shipno': shipnos
    }

def predict_test_data(model, test_loader, device):
    """预测测试数据 - 使用与验证相同的逻辑"""
    model.eval()
    all_predictions = []
    all_shipnos = []
    
    print("🔮 开始预测...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            shipnos = batch['shipno']
            
            # 检查输入数据
            if torch.isnan(sequences).any() or torch.isinf(sequences).any():
                print(f"⚠️ 检测到NaN/Inf输入，进行清理")
                sequences = torch.nan_to_num(sequences, nan=0.0)
                sequences = torch.clamp(sequences, -10, 10)
            
            try:
                # 前向传播 - 与验证逻辑完全一致
                logits = model(sequences, masks)
                
                # 检查输出
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"⚠️ 检测到异常输出，跳过该批次")
                    continue
                
                # 获取预测结果
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_shipnos.extend(shipnos)
                
            except Exception as e:
                print(f"❌ 预测批次出错: {e}")
                continue
    
    return all_predictions, all_shipnos

def main():
    """主预测函数"""
    config = ConfigGridCellPredict()
    
    print("🚀 开始GridCell模型预测...")
    print(f"模型路径: {config.model_path}")
    print(f"测试数据: {config.test_data_dir}")
    print(f"设备: {config.device}")
    
    # 加载测试数据集
    print("\n📂 加载测试数据集...")
    try:
        test_dataset = ShipDatasetWithDiscreteTime(
            data_root=config.test_data_dir,
            max_seqlen=config.max_seqlen,
            min_seqlen=config.min_seqlen,
            is_test=True  # 测试模式，不需要标签
        )
        print(f"✅ 测试样本数: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ 测试数据集加载失败: {e}")
        return
    
    if len(test_dataset) == 0:
        print("❌ 没有加载到测试数据")
        return
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # 加载模型
    model = load_model(config.model_path, config.device)
    if model is None:
        return
    
    model = model.to(config.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 模型参数总数: {total_params:,}")
    
    # 预测
    predictions, shipnos = predict_test_data(model, test_loader, config.device)
    
    if len(predictions) == 0:
        print("❌ 预测失败，没有有效结果")
        return
    
    # 转换预测结果为船舶类型
    predicted_ship_types = [config.ship_types[pred] for pred in predictions]
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'shipno': shipnos,
        'shiptype': predicted_ship_types
    })
    
    # 保存结果
    results.to_csv(config.output_file, index=False, encoding='utf-8')
    
    print(f"\n🎉 预测完成！")
    print(f"📄 结果保存到: {config.output_file}")
    print(f"📊 预测样本数: {len(results)}")
    
    # 显示预测结果分布
    print(f"\n📈 预测结果分布:")
    distribution = results['shiptype'].value_counts()
    for ship_type, count in distribution.items():
        percentage = count / len(results) * 100
        print(f"  {ship_type}: {count} ({percentage:.1f}%)")
    
    # 显示前10个预测结果
    print(f"\n📋 前10个预测结果:")
    print(results.head(10).to_string(index=False))
    
    # 检查是否有重复的shipno
    duplicates = results['shipno'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ 发现 {duplicates} 个重复的shipno")
    else:
        print(f"✅ 所有shipno都是唯一的")

if __name__ == "__main__":
    main()