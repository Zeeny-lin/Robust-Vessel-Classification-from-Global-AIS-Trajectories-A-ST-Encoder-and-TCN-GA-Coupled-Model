import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class GeoPositionEncoder:
    """
    地理坐标编码器，将经纬度转换为高维空间特征
    """

    def __init__(self, frequency_num=16, max_radius=10000, min_radius=10):
        """
        Args:
            frequency_num: 使用的不同频率正弦波数量
            max_radius: 最大处理半径(单位：公里)
            min_radius: 最小处理半径
        """
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self._cal_freq_list()

    def _cal_freq_list(self):
        """计算频率列表"""
        if self.frequency_num == 1:
            self.freq_list = np.array([1.0 / self.max_radius])
        else:
            self.freq_list = np.logspace(
                np.log10(1.0 / self.max_radius),
                np.log10(1.0 / self.min_radius),
                num=self.frequency_num
            )

    def encode(self, coords):
        """
        编码经纬度坐标
        Args:
            coords: 形状为 [n_points, 2] 的数组，每行是[经度, 纬度]
        Returns:
            编码后的特征，形状为 [n_points, frequency_num*6]
        """
        # 转换为弧度
        coords_rad = np.deg2rad(coords)
        lon, lat = coords_rad[:, 0], coords_rad[:, 1]

        # 计算各频率分量
        encoded_features = []
        for freq in self.freq_list:
            lon_scaled = lon * freq
            lat_scaled = lat * freq

            # 基本三角函数特征
            lon_sin = np.sin(lon_scaled)
            lon_cos = np.cos(lon_scaled)
            lat_sin = np.sin(lat_scaled)
            lat_cos = np.cos(lat_scaled)

            # 组合特征
            features = np.column_stack([
                lat_sin, lat_cos,
                lon_sin, lon_cos,
                lat_cos * lon_cos,
                lat_cos * lon_sin
            ])
            encoded_features.append(features)

        # 合并所有频率的特征
        return np.concatenate(encoded_features, axis=1)

class Dataset_Ship_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features=['lat', 'lon', 'sog', 'cog'],
                 scale=True, timeenc=0, freq='h',
                 max_seq_len=512, min_seq_len=50,
                 geo_encode=True, geo_freq_num=4):
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
            self.geo_encoder = GeoPositionEncoder(
                frequency_num=geo_freq_num,
                max_radius=10000,  # 可根据数据范围调整
                min_radius=10
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
        location_sequences = []  # 存储经纬度序列

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

                        # # 提取特征数据
                        # # feature_data = ship_data[self.features].values
                        # base_features = [f for f in self.features if f not in ['lat', 'lon']]
                        # feature_data = ship_data[base_features].values if base_features else None
                        #
                        # # 处理地理坐标
                        # if 'lat' in self.features and 'lon' in self.features:
                        #     geo_coords = ship_data[['lon', 'lat']].values  # 注意顺序:经度,纬度
                        #
                        #     if self.geo_encode:
                        #         # 地理编码
                        #         geo_features = self.geo_encoder.encode(geo_coords)
                        #     else:
                        #         # 直接使用原始坐标
                        #         geo_features = geo_coords
                        #
                        #     # 合并特征
                        #     if feature_data is not None:
                        #         feature_data = np.concatenate([geo_features, feature_data], axis=1)
                        #     else:
                        #         feature_data = geo_features

                        # 处理长序列：滑动窗口切分
                        feature_data = ship_data[self.features].values

                        # 提取经纬度数据用于位置编码
                        if 'lat' in self.features and 'lon' in self.features:
                            lat_idx = self.features.index('lat')
                            lon_idx = self.features.index('lon')
                            lat_lon_data = feature_data[:, [lat_idx, lon_idx]]
                        else:
                            lat_lon_data = None

                        if len(feature_data) > self.seq_len:
                            # 使用滑动窗口，步长为seq_len的一半
                            step = self.seq_len // 2
                            for start_idx in range(0, len(feature_data) - self.seq_len + 1, step):
                                seq = feature_data[start_idx:start_idx + self.seq_len]
                                sequences.append(seq)
                                labels.append(ship_type)
                                if lat_lon_data is not None:
                                    lat_lon_seq = lat_lon_data[start_idx:start_idx + self.seq_len]
                                    location_sequences.append(lat_lon_seq)
                        else:
                            # 短序列：填充到固定长度
                            seq = self._pad_sequence(feature_data, self.seq_len)
                            sequences.append(seq)
                            labels.append(ship_type)
                            if lat_lon_data is not None:
                                lat_lon_seq = self._pad_sequence(lat_lon_data, self.seq_len)
                                location_sequences.append(lat_lon_seq)

                except Exception as e:
                    print(f"Error loading {csv_file}: {str(e)}")
                    continue

        print(f"Loaded {len(sequences)} sequences from {len(ship_types)} ship types")
        print(f"Ship types: {ship_types}")

        # 转换为numpy数组
        self.data_x = np.array(sequences, dtype=np.float32)

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
        # 位置编码
        location_embedding = None
        if self.geo_encode:
            try:
                lat_lon_seq = torch.FloatTensor(self.location_data[index])  # [seq_len, 2]
                location_embedding = self.location_encoder.encode(lat_lon_seq)  # [seq_len, embed_dim]
            except Exception as e:
                print(f"Warning: Location encoding failed for sample {index}: {str(e)}")
                location_embedding = None

        if location_embedding is not None:
            return seq_x, label, location_embedding
        else:
            return seq_x, label

        return seq_x, label

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
    features = ['lat', 'lon', 'sog', 'cog','delta_time','sin_hour']  # 可以根据需要调整

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

