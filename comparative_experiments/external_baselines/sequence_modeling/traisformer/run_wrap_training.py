#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动 Wrap 预训练模型训练
一键运行脚本
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查环境"""
    print("🔍 检查运行环境...")
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 检查 PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['numpy', 'sklearn', 'matplotlib', 'seaborn', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            return False
    
    return True


def check_data_and_model():
    """检查数据和模型文件"""
    print("\n📁 检查文件...")
    
    # 检查数据目录
    data_root = r"G:/test_filled_1enhance"
    if os.path.exists(data_root):
        print(f"✅ 数据目录存在: {data_root}")
        
        # 统计数据文件
        pkl_files = list(Path(data_root).rglob("*.pkl"))
        print(f"📊 找到 {len(pkl_files)} 个 .pkl 文件")
    else:
        print(f"❌ 数据目录不存在: {data_root}")
        print("💡 请修改 WrapTrainingConfig.data_root 为正确路径")
    
    # 检查 Wrap 预训练模型
    wrap_model_candidates = [
        r'F:\TrAISformer-main\TorchSpatial-main\models\wrap\model_birdsnap_ebird_meta_wrap_inception_v3_0.0050_64_0.0001000_1_512.pth.tar',
        r'F:\TrAISformer-main\TorchSpatial-main\pre_trained_models\wrap\model_birdsnap_ebird_meta_wrap_inception_v3_0.0050_64_0.0001000_1_512.pth.tar'
    ]
    
    wrap_found = False
    for candidate in wrap_model_candidates:
        if os.path.exists(candidate):
            print(f"✅ 找到 Wrap 预训练模型: {candidate}")
            wrap_found = True
            break
    
    if not wrap_found:
        print("❌ 未找到 Wrap 预训练模型")
        print("💡 请检查 TorchSpatial 安装路径")
    
    # 检查 TorchSpatial 路径
    torchspatial_path = r'F:\TrAISformer-main\TorchSpatial-main\main'
    if os.path.exists(torchspatial_path):
        print(f"✅ TorchSpatial 路径存在: {torchspatial_path}")
    else:
        print(f"❌ TorchSpatial 路径不存在: {torchspatial_path}")
        print("💡 请修改代码中的 TorchSpatial 路径")
    
    return wrap_found


def run_training():
    """运行训练"""
    print("\n🚀 启动 Wrap 预训练模型训练...")
    
    try:
        from train_wrap_ship_classifier import main
        main()
    except ImportError as e:
        print(f"❌ 导入训练模块失败: {e}")
        print("💡 请确保 train_wrap_ship_classifier.py 在当前目录")
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🌍 Wrap 预训练船舶分类器 - 一键启动")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请安装缺失的依赖包")
        return
    
    # 检查文件
    if not check_data_and_model():
        print("\n❌ 文件检查失败，请检查数据和模型路径")
        return
    
    print("\n✅ 所有检查通过，准备开始训练...")
    
    # 确认启动
    response = input("\n是否开始训练？(y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        run_training()
    else:
        print("❌ 用户取消训练")


if __name__ == "__main__":
    main()