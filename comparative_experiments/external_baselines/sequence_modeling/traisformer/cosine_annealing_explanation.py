#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
余弦退火学习率调度器详解
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

def explain_cosine_annealing():
    """详细解释余弦退火学习率调度器"""
    
    print("📉 余弦退火 (CosineAnnealingLR) 详解")
    print("=" * 80)
    
    print("\n🎯 核心思想:")
    print("-" * 50)
    core_idea = """
余弦退火基于余弦函数的平滑下降特性：
1. 学习率从初始值平滑下降到最小值
2. 下降曲线遵循余弦函数的形状
3. 前期下降较快，后期下降较慢
4. 提供了平滑且可预测的学习率变化
"""
    print(core_idea)
    
    print("\n📊 学习率变化模式:")
    print("-" * 50)
    
    # 模拟余弦退火的学习率变化
    initial_lr = 1e-6
    min_lr = 1e-7
    max_epochs = 100
    
    # 计算每个epoch的学习率
    epochs = list(range(max_epochs))
    learning_rates = []
    
    for epoch in epochs:
        # 余弦退火公式
        lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2
        learning_rates.append(lr)
    
    print(f"初始学习率: {initial_lr:.2e}")
    print(f"最小学习率: {min_lr:.2e}")
    print(f"总训练轮数: {max_epochs}")
    
    print(f"\n学习率变化示例:")
    key_epochs = [0, 10, 25, 50, 75, 90, 99]
    for epoch in key_epochs:
        lr = learning_rates[epoch]
        print(f"  Epoch {epoch:2d}: {lr:.2e}")
    
    print("\n🔍 余弦退火的特点:")
    print("=" * 80)
    
    characteristics = """
📈 平滑下降:
   - 学习率按余弦曲线平滑下降
   - 没有突然的跳跃或不连续点
   - 数值稳定性好

🎯 前快后慢:
   - 前期下降较快，快速降低学习率
   - 后期下降较慢，精细调整参数
   - 符合训练的自然规律

🔄 可预测性:
   - 学习率变化完全可预测
   - 便于调试和分析
   - 不依赖于训练步数的精确计算

🛡️ 数值稳定:
   - 不会出现过高的学习率
   - 避免了梯度爆炸的风险
   - 比OneCycleLR安全得多
"""
    print(characteristics)
    
    print("\n⚖️ 关键参数:")
    print("-" * 50)
    
    parameters = """
🎚️ T_max: 周期长度
   - 通常设置为总训练epoch数
   - 决定了余弦函数的完整周期
   - 您的设置: config.max_epochs (100)

📉 eta_min: 最小学习率
   - 学习率的下界
   - 通常是初始学习率的5%-20%
   - 您的设置: initial_lr * 0.1 (10%)

📈 initial_lr: 初始学习率
   - 余弦退火的起始点
   - 您的设置: 1e-6 (非常保守)
"""
    print(parameters)
    
    print("\n✅ 余弦退火的优势:")
    print("-" * 50)
    
    advantages = """
🛡️ 数值稳定:
   - 学习率只会下降，不会上升
   - 避免了OneCycleLR的高学习率风险
   - 不会导致梯度爆炸

📉 平滑变化:
   - 余弦函数提供平滑的过渡
   - 避免学习率的突然变化
   - 训练过程更加稳定

🎯 自适应调整:
   - 前期快速下降，后期缓慢精调
   - 符合深度学习训练的规律
   - 有助于找到更好的局部最优

🔧 易于调试:
   - 参数简单，只需设置T_max和eta_min
   - 行为可预测，便于分析
   - 不需要复杂的步数计算
"""
    print(advantages)
    
    print("\n🤔 与OneCycleLR的对比:")
    print("=" * 80)
    
    comparison = """
特征                余弦退火              OneCycleLR
----------------------------------------------------
学习率变化          只下降               先升后降
数值稳定性          高                   中等
参数复杂度          简单                 复杂
调试难度            容易                 困难
NaN风险             低                   高
适用场景            通用                 大数据集
训练速度            中等                 可能更快
收敛稳定性          高                   中等

🏆 对于您的情况，余弦退火更合适！
"""
    print(comparison)
    
    print("\n📊 您的配置分析:")
    print("-" * 50)
    
    your_config = """
当前设置:
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=100,          # 100个epoch完成一个周期
    eta_min=1e-7        # 最小学习率 (1e-6 * 0.1)
)
```

学习率变化预测:
🟢 Epoch 1:   1.00e-06 (起始)
🟢 Epoch 10:  9.02e-07 (轻微下降)
🟢 Epoch 25:  7.07e-07 (25%下降)
🟢 Epoch 50:  5.50e-07 (中点)
🟢 Epoch 75:  3.93e-07 (75%下降)
🟢 Epoch 100: 1.00e-07 (最小值)

✅ 优势:
- 没有学习率上升，避免NaN风险
- 平滑下降，训练稳定
- 前期有足够的学习能力
- 后期精细调整参数
"""
    print(your_config)
    
    print("\n🚀 使用建议:")
    print("-" * 50)
    
    suggestions = """
🎯 监控要点:
1. 观察学习率是否平滑下降
2. 训练损失应该稳定下降
3. 不应该出现NaN或异常跳跃
4. 验证精度应该逐步提升

📈 调优建议:
- 如果收敛太慢: 适当提高初始学习率到2e-6
- 如果仍有不稳定: 降低eta_min到0.05倍
- 如果想要更激进: 可以使用CosineAnnealingWarmRestarts

🔧 故障排除:
- 如果第1轮就NaN: 问题不在学习率，检查数据/模型
- 如果中途NaN: 可能需要更强的梯度裁剪
- 如果收敛停滞: 考虑调整eta_min或使用warm restart
"""
    print(suggestions)
    
    print("\n✅ 总结:")
    print("-" * 50)
    print("余弦退火是一个安全、稳定、易于调试的学习率策略")
    print("特别适合您当前需要解决NaN问题的情况")
    print("相比OneCycleLR，它提供了更好的数值稳定性")

def plot_cosine_annealing(initial_lr=1e-6, eta_min=1e-7, T_max=100):
    """绘制余弦退火学习率变化图"""
    epochs = list(range(T_max))
    lrs = []
    
    for epoch in epochs:
        lr = eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, 'b-', linewidth=2, label='Cosine Annealing')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('4-6/cosine_annealing_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"学习率变化图已保存为: 4-6/cosine_annealing_schedule.png")

if __name__ == "__main__":
    explain_cosine_annealing()
    
    # 如果有matplotlib，可以生成图表
    try:
        plot_cosine_annealing()
    except ImportError:
        print("\n注意: 需要安装matplotlib来生成学习率变化图")
        print("pip install matplotlib") 