from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Ship_Classification
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

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
    features = ['lat', 'lon', 'sog', 'cog']  # 可以根据需要调整

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

