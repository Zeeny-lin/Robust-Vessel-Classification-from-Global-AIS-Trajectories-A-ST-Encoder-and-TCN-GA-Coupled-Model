#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试时间特征模型是否能正常运行
"""

import torch
import numpy as np
from config_ship_classification_with_time import ShipClassificationConfigWithTime
from train_ship_classification_with_time import TrAISformerClassifierWithTime

def test_model():
    print("测试时间特征模型...")
    
    # 创建配置
    config = ShipClassificationConfigWithTime()
    config.device = "cpu"  # 使用CPU进行测试
    
    print(f"配置信息:")
    print(f"  特征维度: {config.feature_dim}")
    print(f"  模式: {config.mode}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  时间特征尺寸: interval={config.time_interval_size}, window={config.time_window_size}")
    
    # 创建模型
    model = TrAISformerClassifierWithTime(config, num_classes=4)
    model.eval()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    feature_dim = 6
    
    # 创建标准化的输入数据 [0, 1)
    x = torch.rand(batch_size, seq_len, feature_dim)
    
    # 确保数据在正确范围内
    x[:, :, 0] = torch.clamp(x[:, :, 0], 0, 0.9999)  # lat
    x[:, :, 1] = torch.clamp(x[:, :, 1], 0, 0.9999)  # lon
    x[:, :, 2] = torch.clamp(x[:, :, 2], 0, 0.9999)  # sog
    x[:, :, 3] = torch.clamp(x[:, :, 3], 0, 0.9999)  # cog
    x[:, :, 4] = torch.clamp(x[:, :, 4], 0, 0.9999)  # time_interval
    x[:, :, 5] = torch.clamp(x[:, :, 5], 0, 0.9999)  # time_window
    
    # 创建mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"输入数据形状: {x.shape}")
    print(f"输入数据范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Mask形状: {mask.shape}")
    
    try:
        # 前向传播
        with torch.no_grad():
            logits = model(x, mask)
        
        print(f"输出logits形状: {logits.shape}")
        print(f"输出logits: {logits}")
        
        # 检查预测
        predictions = torch.argmax(logits, dim=1)
        print(f"预测结果: {predictions}")
        
        print("✅ 模型测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 模型可以正常运行！")
    else:
        print("\n💥 模型仍有问题，需要进一步调试。") 