import torch

class ShipClassificationConfigSimple:
    # 数据路径配置
    data_root = "G:/data_wash_enhanced"  # 您可以修改为实际路径
    
    # 数据配置
    num_classes = 4
    max_seqlen = 650   # 与原始配置一致
    min_seqlen = 50     # 设置为50，基本不过滤
    
    # 船舶类型映射
    ship_types = ['集装箱船', '干散货船', '渔船', '油船']
    
    # 模型配置 - 与原始配置完全一致
    n_embd = 512
    n_head = 16
    n_layer = 6
    pooling_strategy = 'attention'  # 'mean', 'max', 'attention', 'last'
    dropout = 0.1
    
    # 训练配置 - 与原始文件相似但更保守
    batch_size = 16   # 与原始文件相同
    gradient_accumulation_steps = 4   # 与原始文件相同
    learning_rate = 1e-4  # 与原始文件相同的学习率 (最大学习率)
    min_learning_rate_ratio = 0.05  # 最小学习率比例 (最小学习率 = learning_rate * min_learning_rate_ratio)
    max_epochs = 100
    weight_decay = 0.01   # 与原始文件相同
    
    # 混合精度训练
    mixed_precision = False  # 暂时关闭混合精度，避免数值不稳定
    
    # 早停机制
    patience = 15
    
    # 内存优化选项
    gradient_checkpointing = False  # 暂时关闭，因为当前模型不支持
    
    # 数据增强配置
    augmentation = True         # 是否使用数据增强
    noise_std = 0.01           # 高斯噪声标准差
    dropout_prob = 0.1         # 随机遮蔽概率
    time_shift_ratio = 0.1     # 时间偏移比例
    
    # 损失函数配置
    use_focal_loss = True      # 使用Focal Loss
    focal_alpha = None         # 类别权重，None表示自动计算
    focal_gamma = 2            # Focal Loss gamma参数
    
    # 验证配置
    val_ratio = 0.05           # 验证集比例
    
    # 输出配置
    output_dir = "G:/out-simple"  # 输出目录，您可以修改
    save_every_n_epochs = 5    # 每N轮保存一次检查点
    
    # 日志配置
    log_every_n_steps = 50     # 每N步记录一次日志
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __str__(self):
        """打印配置信息"""
        config_str = "ShipClassificationConfigSimple:\n"
        config_str += f"  数据路径: {self.data_root}\n"
        config_str += f"  特征维度: 6 (lat, lon, sog, cog, time_interval, time_window)\n"
        config_str += f"  序列长度: {self.min_seqlen}-{self.max_seqlen}\n"
        config_str += f"  模型参数: {self.n_layer}层, {self.n_head}头, {self.n_embd}维\n"
        config_str += f"  训练参数: bs={self.batch_size}, acc_steps={self.gradient_accumulation_steps}, lr={self.learning_rate}\n"
        config_str += f"  池化策略: {self.pooling_strategy}\n"
        config_str += f"  输出目录: {self.output_dir}\n"
        config_str += f"  设备: {self.device}\n"
        return config_str 