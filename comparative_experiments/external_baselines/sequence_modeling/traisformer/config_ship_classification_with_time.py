import torch

class ShipClassificationConfigWithTime:
    # 数据路径配置
    data_root = r"G:\data_wash_enhanced"  # 使用增强后的数据（您可以改为原数据路径）
    
    # 数据配置
    num_classes = 4
    max_seqlen = 650    # 最大序列长度
    min_seqlen = 50     # 最小序列长度，过滤太短的轨迹
    
    # 船舶类型映射
    ship_types = ['集装箱船', '干散货船', '渔船', '油船']
    
    # 时间特征配置
    feature_dim = 6  # 6个特征：lat, lon, sog, cog, time_interval, time_window
    
    # 基础特征尺寸（用于transformer embedding）
    lat_size = 250
    lon_size = 270
    sog_size = 30
    cog_size = 72
    
    # 时间特征尺寸
    time_interval_size = 8000   # 时间间隔特征的离散化大小（覆盖0-8000秒，约2.2小时）
    time_window_size = 744      # 时间窗口特征大小（31天*24小时）
    
    # 基本特征embedding维度
    n_lat_embd = 256
    n_lon_embd = 256
    n_sog_embd = 128
    n_cog_embd = 128
    
    # 时间特征embedding维度
    n_time_interval_embd = 128  # 时间间隔特征embedding维度（增加以处理更大的特征空间）
    n_time_window_embd = 128    # 时间窗口特征embedding维度
    
    # 模型配置
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd + n_time_interval_embd + n_time_window_embd  # 总embedding维度
    n_head = 16         # attention头数（与原版一致）
    n_layer = 6         # transformer层数
    dropout = 0.1       # dropout率
    
    # transformer模型配置
    mode = "mlp"        # 使用mlp模式，直接处理实值特征
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # 训练配置
    batch_size = 64             # 批大小（与原版一致，如果内存不足可改为8）
    gradient_accumulation_steps = 2  # 梯度累积步数
    learning_rate = 1e-4        # 学习率
    max_epochs = 100            # 最大训练轮数（与原版一致）
    weight_decay = 0.01         # 权重衰减
    
    # 学习率调度
    lr_decay = True
    warmup_ratio = 0.1          # 学习率预热比例
    
    # 混合精度训练
    mixed_precision = True      # 启用混合精度
    
    # 早停机制
    patience = 15               # 早停耐心值（与原版一致）
    
    # 内存优化选项
    gradient_checkpointing = True   # 启用梯度检查点节省内存
    
    # 数据增强配置
    augmentation = True         # 是否使用数据增强
    noise_std = 0.01           # 高斯噪声标准差
    dropout_prob = 0.1         # 随机遮蔽概率
    
    # 损失函数配置
    use_focal_loss = True      # 使用Focal Loss
    focal_alpha = None         # 类别权重，None表示自动计算
    focal_gamma = 2            # Focal Loss gamma参数
    
    # 验证配置
    val_ratio = 0.15           # 验证集比例
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 输出配置
    output_dir = "./results"
    save_every_n_epochs = 5    # 每N轮保存一次检查点
    
    # 日志配置
    log_every_n_steps = 50     # 每N步记录一次日志
    
    # 坐标范围配置（根据实际数据调整）
    lat_min = 0.0
    lat_max = 1.0
    lon_min = 0.0
    lon_max = 1.0
    
    def __str__(self):
        """打印配置信息"""
        config_str = "ShipClassificationConfigWithTime:\n"
        config_str += f"  数据路径: {self.data_root}\n"
        config_str += f"  特征维度: {self.feature_dim}\n"
        config_str += f"  序列长度: {self.min_seqlen}-{self.max_seqlen}\n"
        config_str += f"  模型参数: {self.n_layer}层, {self.n_head}头, {self.n_embd}维\n"
        config_str += f"  训练参数: bs={self.batch_size}, acc_steps={self.gradient_accumulation_steps}, lr={self.learning_rate}\n"
        config_str += f"  设备: {self.device}\n"
        return config_str 