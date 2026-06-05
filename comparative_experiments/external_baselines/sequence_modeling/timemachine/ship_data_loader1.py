import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class Dataset_Ship_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features=['lat', 'lon', 'sog', 'cog'],
                 scale=True, timeenc=0, freq='h',
                 max_seq_len=512, min_seq_len=50):
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
        from torchspatial.models import Space2Vec_grid
        import torch
        self.loc_encoder = Space2Vec_grid(
            in_dim=2,  # 输入为 (lat, lon)
            out_dim=32,  # 输出维度，可根据模型调整
            num_scales=6  # 多尺度频率数量
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

                        # 提取特征数据
                        feature_data = ship_data[self.features].values

                        # 如果包含 lat/lon，用 GeoEncoder 进行编码
                        if 'lat' in self.features and 'lon' in self.features:
                            latlon_idx = [self.features.index('lat'), self.features.index('lon')]
                            latlon_data = feature_data[:, latlon_idx]
                            other_feature_indices = [i for i in range(len(self.features)) if i not in latlon_idx]
                            other_data = feature_data[:, other_feature_indices] if other_feature_indices else None

                            # 将 latlon 转为 tensor 并编码
                            latlon_tensor = torch.tensor(latlon_data, dtype=torch.float32)
                            encoded_geo = self.loc_encoder(latlon_tensor).numpy()  # 输出 shape: (T, geo_out_dim)

                            # 重新组合特征
                            if other_data is not None:
                                feature_data = np.concatenate([encoded_geo, other_data], axis=1)
                            else:
                                feature_data = encoded_geo

                        # 过滤过短的序列
                        if len(feature_data) < self.min_seq_len:
                            continue

                        # 处理长序列：滑动窗口切分
                        if len(feature_data) > self.seq_len:
                            # 使用滑动窗口，步长为seq_len的一半
                            step = self.seq_len // 2
                            for start_idx in range(0, len(feature_data) - self.seq_len + 1, step):
                                seq = feature_data[start_idx:start_idx + self.seq_len]
                                sequences.append(seq)
                                labels.append(ship_type)
                        else:
                            # 短序列：填充到固定长度
                            seq = self._pad_sequence(feature_data, self.seq_len)
                            sequences.append(seq)
                            labels.append(ship_type)

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
        lat_lon = self.data_x[index][:, :2]
        # 编码处理
        encoded = self.loc_encoder(lat_lon)
        # 替换原始坐标
        self.data_x[index][:, :2] = encoded

        seq_x = self.data_x[index]  # [seq_len, n_features]
        label = self.data_y[index]  # 标量标签

        # 转换为tensor
        seq_x = torch.FloatTensor(seq_x)
        label = torch.LongTensor([label])

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

