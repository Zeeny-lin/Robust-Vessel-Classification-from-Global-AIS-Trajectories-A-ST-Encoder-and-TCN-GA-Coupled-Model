import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from main.SpatialRelationEncoder import GridCellSpatialRelationLocationEncoder

warnings.filterwarnings('ignore')

# class GeoPositionEncoder:
#     """
#     地理坐标编码器，将经纬度转换为高维空间特征
#     """
#
#     def __init__(self, frequency_num=16, max_radius=10000, min_radius=10):
#         """
#         Args:
#             frequency_num: 使用的不同频率正弦波数量
#             max_radius: 最大处理半径(单位：公里)
#             min_radius: 最小处理半径
#         """
#         self.frequency_num = frequency_num
#         self.max_radius = max_radius
#         self.min_radius = min_radius
#         self._cal_freq_list()
#
#     def _cal_freq_list(self):
#         """计算频率列表"""
#         if self.frequency_num == 1:
#             self.freq_list = np.array([1.0 / self.max_radius])
#         else:
#             self.freq_list = np.logspace(
#                 np.log10(1.0 / self.max_radius),
#                 np.log10(1.0 / self.min_radius),
#                 num=self.frequency_num
#             )
#
#     def encode(self, coords):
#         """
#         编码经纬度坐标
#         Args:
#             coords: 形状为 [n_points, 2] 的数组，每行是[经度, 纬度]
#         Returns:
#             编码后的特征，形状为 [n_points, frequency_num*6]
#         """
#         # 转换为弧度
#         coords_rad = np.deg2rad(coords)
#         lon, lat = coords_rad[:, 0], coords_rad[:, 1]
#
#         # 计算各频率分量
#         encoded_features = []
#         for freq in self.freq_list:
#             lon_scaled = lon * freq
#             lat_scaled = lat * freq
#
#             # 基本三角函数特征
#             lon_sin = np.sin(lon_scaled)
#             lon_cos = np.cos(lon_scaled)
#             lat_sin = np.sin(lat_scaled)
#             lat_cos = np.cos(lat_scaled)
#
#             # 组合特征
#             features = np.column_stack([
#                 lat_sin, lat_cos,
#                 lon_sin, lon_cos,
#                 lat_cos * lon_cos,
#                 lat_cos * lon_sin
#             ])
#             encoded_features.append(features)
#
#         # 合并所有频率的特征
#         return np.concatenate(encoded_features, axis=1)


class Dataset_Ship_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features=['lat', 'lon', 'sog', 'cog'],
                 scale=True, timeenc=0, freq='h',
                 max_seq_len=512, min_seq_len=50,
                 geo_encode=True, geo_freq_num=8,
                 spa_embed_dim=64, device="cuda"):
        """
        船舶轨迹分类数据集

        Args:
            root_path: 数据根目录，包含train/test/val文件夹
            flag: 'train', 'test', 'val'
            size: [seq_len] 序列长度
            features: 使用的特征列名
            scale: 是否标准化
            max_seq_len: 最大序列长度
            min_seq_len: 最小序列长度（过滤短序列）
        """
        self.geo_encode = geo_encode
        self.geo_freq_num = geo_freq_num
        if self.geo_encode:
            self.geo_encoder = GridCellSpatialRelationLocationEncoder(
                spa_embed_dim=spa_embed_dim,
                coord_dim=2,
                frequency_num=8,
                max_radius=10000,
                min_radius=10,
                freq_init="geometric",
                device=device,
                ffn_act="relu",
                ffn_num_hidden_layers=1,
                ffn_dropout_rate=0.5,
                ffn_hidden_dim=256,
                ffn_use_layernormalize=True,
                ffn_skip_connection=True,
                ffn_context_str="GridCellSpatialRelationEncoder"
            )

        # 序列长度设置
        if size is None:
            self.seq_len = 256  # 默认序列长度
        else:
            self.seq_len = size[0]

        assert flag in ['train', 'test', 'val']

        self.flag = flag
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.root_path = root_path

        # 读取数据
        self.__read_data__()

    def __read_data__(self):
        """读取船舶轨迹数据"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 构建数据路径
        data_path = os.path.join(self.root_path, self.flag)

        sequences = []
        labels = []
        ship_types = []
        file_names = []
        location_sequences = []

        # 遍历每个船舶类型文件夹
        for ship_type in os.listdir(data_path):
            ship_type_path = os.path.join(data_path, ship_type)
            if not os.path.isdir(ship_type_path):
                continue

            ship_types.append(ship_type)
            print(f"Loading {ship_type} data from {ship_type_path}...")

            # 遍历该类型下的所有CSV文件
            csv_files = [f for f in os.listdir(ship_type_path) if f.endswith('.csv')]

            for csv_file in csv_files:
                csv_path = os.path.join(ship_type_path, csv_file)

                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)

                    # 检查必要的列是否存在
                    required_cols = ['shipno', 'postime'] + self.features
                    if not all(col in df.columns for col in required_cols):
                        print(f"Warning: {csv_file} missing required columns, skipping...")
                        continue

                    # 按船舶编号分组处理轨迹
                    for shipno, ship_data in df.groupby('shipno'):
                        # 按时间排序
                        ship_data = ship_data.sort_values('postime')

                        if 'lat' in self.features and 'lon' in self.features:
                            lat_idx = self.features.index('lat')
                            lon_idx = self.features.index('lon')

                            # 提取经纬度数据用于地理编码
                            lat_lon_data = ship_data[['lat', 'lon']].values

                            # 提取除经纬度外的其他特征
                            other_features = [f for f in self.features if f not in ['lat', 'lon']]
                            if other_features:
                                feature_data = ship_data[other_features].values
                            else:
                                feature_data = np.empty((len(ship_data), 0))  # 空特征数组
                        else:
                            # 没有经纬度，使用所有特征
                            feature_data = ship_data[self.features].values
                            lat_lon_data = None

                        # 过滤过短的序列
                        if len(feature_data) < self.min_seq_len:
                            continue

                        # 处理长序列：滑动窗口切分
                        if len(feature_data) > self.seq_len:
                            # 使用滑动窗口，步长为seq_len的一半
                            seq = feature_data[:self.seq_len]
                            sequences.append(seq)
                            labels.append(ship_type)
                            file_name = os.path.splitext(csv_file)[0]
                            file_names.append(f"{file_name}")
                            if lat_lon_data is not None:
                                lat_lon_seq = lat_lon_data[:self.seq_len]
                                location_sequences.append(lat_lon_seq)
                            # step = self.seq_len
                            # for start_idx in range(0, len(feature_data) - self.seq_len + 1, step):
                            #     seq = feature_data[start_idx:start_idx + self.seq_len]
                            #     sequences.append(seq)
                            #     labels.append(ship_type)
                            #     file_name = os.path.splitext(csv_file)[0]
                            #     file_names.append(f"{file_name}")
                            #     # file_names.append(f"{csv_file}")

                        else:
                            # 短序列：填充到固定长度
                            seq = self._pad_sequence(feature_data, self.seq_len)
                            sequences.append(seq)
                            labels.append(ship_type)
                            file_name = os.path.splitext(csv_file)[0]
                            file_names.append(f"{file_name}")
                            # file_names.append(f"{csv_file}")
                            if lat_lon_data is not None:
                                lat_lon_seq = self._pad_sequence(lat_lon_data, self.seq_len)
                                # lat_lon_seq = lat_lon_data[:self.seq_len]
                                location_sequences.append(lat_lon_seq)

                except Exception as e:
                    print(f"Error loading {csv_file}: {str(e)}")
                    continue

        print(f"Loaded {len(sequences)} sequences from {len(ship_types)} ship types")
        print(f"Ship types: {ship_types}")
        self.file_names = file_names

        # 转换为numpy数组
        self.data_x = np.array(sequences, dtype=np.float32)
        if np.isnan(self.data_x).any() or np.isinf(self.data_x).any():
            print("Warning: Data contains NaN or Inf values!")
            # 处理方式：填充或删除异常值
            self.data_x = np.nan_to_num(self.data_x, nan=0.0, posinf=1e5, neginf=-1e5)

        # 编码标签
        self.label_encoder.fit(labels)
        self.data_y = self.label_encoder.transform(labels)
        self.num_classes = len(self.label_encoder.classes_)

        print(f"Data shape: {self.data_x.shape}")
        print(f"Labels shape: {self.data_y.shape}")
        print(f"Number of classes: {self.num_classes}")

        # 数据标准化
        if self.scale:
            # 重塑数据以便标准化
            original_shape = self.data_x.shape
            self.data_x_reshaped = self.data_x.reshape(-1, self.data_x.shape[-1])

            if self.flag == 'train':
                # 只在训练集上拟合scaler
                self.scaler.fit(self.data_x_reshaped)

            # 应用标准化
            if not hasattr(self.scaler, 'mean_'):  # 检查是否已经拟合过
                self.scaler.fit(self.data_x_reshaped)  # 先拟合
            self.data_x_scaled = self.scaler.transform(self.data_x_reshaped)
            self.data_x = self.data_x_scaled.reshape(original_shape)

    def _pad_sequence(self, sequence, target_length):
        """填充序列到目标长度"""
        seq_len = len(sequence)
        if seq_len >= target_length:
            return sequence[:target_length]

        # 使用重复填充
        repeat_times = target_length // seq_len
        remainder = target_length % seq_len

        padded_seq = np.tile(sequence, (repeat_times, 1))
        if remainder > 0:
            padded_seq = np.vstack([padded_seq, sequence[:remainder]])

        return padded_seq

    def __getitem__(self, index):
        """获取单个样本"""
        seq_x = self.data_x[index]  # [seq_len, n_features]
        label = self.data_y[index]  # 标量标签

        # 转换为tensor
        seq_x = torch.FloatTensor(seq_x)
        label = torch.LongTensor([label])
        file_name = self.file_names[index]
        if self.geo_encode and self.location_sequences is not None:
            # 获取地理坐标序列 [seq_len, 2]
            location_seq = self.location_sequences[index]

            # 转换为GridCellSpatialRelationLocationEncoder需要的格式
            # 从 [seq_len, 2] 转换为 [1, seq_len, 2] (添加batch维度)
            coords_input = np.expand_dims(location_seq, axis=0)

            # 使用GridCellSpatialRelationLocationEncoder进行编码
            # 输入格式: [batch_size, seq_len, 2]
            # 输出格式: [batch_size, seq_len, spa_embed_dim]
            location_embedding = self.geo_encoder(coords_input)

            # 移除batch维度，返回 [seq_len, spa_embed_dim]
            if isinstance(location_embedding, torch.Tensor):
                location_embedding = location_embedding.squeeze(0)
            else:
                location_embedding = torch.FloatTensor(location_embedding).squeeze(0)

        return seq_x, label,file_name,location_embedding

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        """反标准化"""
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data

    def get_label_names(self):
        """获取类别名称"""
        return self.label_encoder.classes_


def ship_data_provider(args, flag):
    """
    船舶分类数据提供器
    """
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:  # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # 使用的特征列
    features = ['lat', 'lon', 'sog', 'cog','delta_time','hour_of_month','sin_timeofday','cos_timeofday']  # 可以根据需要调整

    data_set = Dataset_Ship_Classification(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len],  # 只需要序列长度
        features=features,
        scale=args.scale if hasattr(args, 'scale') else True,
        timeenc=timeenc,
        freq=args.freq,
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 512,
        min_seq_len=args.min_seq_len if hasattr(args, 'min_seq_len') else 50
    )

    print(f"{flag}: {len(data_set)} samples")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader

