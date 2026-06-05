"""
功能   : 轨迹 + 时间特征 训练 + 进度条 + 后处理投票合并 (消融实验 - Extra Trees)
         模型：Extra Trees（极端随机树）
输入   : 从 [lat, lon, sog, cog, delta_h, day_frac] 6维特征中提取更多特征
         新增：保存验证集 TOP-5 模型并做加权投票 + 早停机制 + 特征工程
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm, trange
import heapq
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import joblib
import math

warnings.filterwarnings('ignore')

# -------------- 路径配置 --------------
DATA_ROOT = r'D:\日常项目\25.8论文\data\data\process_seg'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

RESULT_DIR = r'D:\日常项目\25.8论文\result\ExtraTrees_Ablation'
os.makedirs(RESULT_DIR, exist_ok=True)

# 创建当前运行的子文件夹
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, f"run_{current_time}")
os.makedirs(RUN_DIR, exist_ok=True)

# 输出文件路径
LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Oil Tanker']
CLASS_MAP = {
    'Bulk Carrier': 0,
    'Container Ship': 1, 
    'Fishing': 2,
    'Oil Tanker': 3
}

# -------------- 特征工程函数 --------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两个经纬度点之间的哈弗辛距离（公里）"""
    R = 6371  # 地球半径（公里）
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def calculate_trajectory_features(trajectory_data):
    """
    从轨迹数据中提取特征
    输入: trajectory_data (numpy array) - [lat, lon, sog, cog, delta_h, day_frac]
    输出: 特征向量
    """
    if len(trajectory_data) == 0:
        return np.zeros(35)  # 返回零向量如果轨迹为空
    
    # 基本统计特征
    basic_features = []
    
    # 1. 位置相关特征
    lats = trajectory_data[:, 0]
    lons = trajectory_data[:, 1]
    
    # 轨迹长度相关
    if len(lats) > 1:
        total_distance = 0
        distances = []
        for i in range(1, len(lats)):
            dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            distances.append(dist)
            total_distance += dist
        
        basic_features.extend([
            total_distance,  # 总距离
            np.mean(distances) if distances else 0,  # 平均段距离
            np.std(distances) if distances else 0,   # 距离标准差
            np.max(distances) if distances else 0,   # 最大段距离
            np.min(distances) if distances else 0,   # 最小段距离
        ])
    else:
        basic_features.extend([0, 0, 0, 0, 0])
    
    # 2. 速度相关特征 (SOG - Speed Over Ground)
    sogs = trajectory_data[:, 2]
    basic_features.extend([
        np.mean(sogs),      # 平均速度
        np.std(sogs),       # 速度标准差
        np.max(sogs),       # 最大速度
        np.min(sogs),       # 最小速度
        np.median(sogs),    # 速度中位数
        len([s for s in sogs if s > 10]) / len(sogs) if len(sogs) > 0 else 0,  # 高速比例
    ])
    
    # 3. 航向相关特征 (COG - Course Over Ground)
    cogs = trajectory_data[:, 3]
    # 将角度转换为弧度计算统计量
    cog_rad = np.radians(cogs)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    
    mean_sin = np.mean(cog_sin)
    mean_cos = np.mean(cog_cos)
    mean_direction = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
    
    basic_features.extend([
        mean_direction,     # 平均航向
        np.std(cogs),       # 航向变化标准差
        len([c for c in np.diff(cogs) if abs(c) > 30]) / max(len(cogs)-1, 1),  # 大幅转向比例
    ])
    
    # 4. 高度变化特征
    delta_hs = trajectory_data[:, 4]
    basic_features.extend([
        np.mean(delta_hs),      # 平均高度变化
        np.std(delta_hs),       # 高度变化标准差
        np.max(delta_hs),       # 最大高度变化
        np.min(delta_hs),       # 最小高度变化
        np.sum(np.abs(delta_hs)),  # 总高度变化绝对值
    ])
    
    # 5. 时间相关特征
    day_fracs = trajectory_data[:, 5]
    if len(day_fracs) > 1:
        duration = day_fracs[-1] - day_fracs[0]
        time_intervals = np.diff(day_fracs)
        basic_features.extend([
            duration,                           # 轨迹持续时间
            np.mean(time_intervals),            # 平均时间间隔
            np.std(time_intervals),             # 时间间隔标准差
            len(day_fracs) / max(duration, 0.001),  # 采样频率
        ])
    else:
        basic_features.extend([0, 0, 0, 0])
    
    # 6. 轨迹形状特征
    if len(lats) > 2:
        # 起点和终点的直线距离
        start_end_dist = haversine_distance(lats[0], lons[0], lats[-1], lons[-1])
        # 轨迹效率（直线距离/总距离）
        efficiency = start_end_dist / total_distance if total_distance > 0 else 0
        
        # 轨迹弯曲度（基于角度变化）
        direction_changes = []
        for i in range(1, len(cogs)-1):
            change = min(abs(cogs[i] - cogs[i-1]), 360 - abs(cogs[i] - cogs[i-1]))
            direction_changes.append(change)
        
        basic_features.extend([
            start_end_dist,                     # 起点终点直线距离
            efficiency,                         # 轨迹效率
            np.mean(direction_changes) if direction_changes else 0,  # 平均方向变化
            np.std(direction_changes) if direction_changes else 0,   # 方向变化标准差
        ])
    else:
        basic_features.extend([0, 0, 0, 0])
    
    # 7. 统计矩特征
    basic_features.extend([
        len(trajectory_data),                   # 轨迹点数
        np.percentile(sogs, 25) if len(sogs) > 0 else 0,  # 速度25分位数
        np.percentile(sogs, 75) if len(sogs) > 0 else 0,  # 速度75分位数
        np.percentile(cogs, 25) if len(cogs) > 0 else 0,  # 航向25分位数
        np.percentile(cogs, 75) if len(cogs) > 0 else 0,  # 航向75分位数
    ])
    
    return np.array(basic_features)

# -------------- 数据集类 --------------
class ShipTrajectoryFeatureDataset:
    def __init__(self, data_dir):
        self.X = []
        self.y = []
        self.filenames = []
        self.required = ['lat', 'lon', 'sog', 'cog', 'delta_h', 'day_frac']
        
        # 遍历每个船型文件夹
        for ship_type in CLASS_NAMES:
            ship_dir = os.path.join(data_dir, ship_type)
            if not os.path.exists(ship_dir):
                continue
                
            csv_files = glob.glob(os.path.join(ship_dir, '*.csv'))
            print(f"Processing {len(csv_files)} files in {ship_type}...")
            
            for csv_file in tqdm(csv_files, desc=f"Extracting features - {ship_type}"):
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.lower() for c in df.columns]
                    
                    # 确保所有必需列都存在
                    for col in self.required:
                        if col not in df.columns:
                            df[col] = 0.0
                    
                    trajectory = df[self.required].astype(float).values
                    if len(trajectory) > 0:
                        # 提取特征
                        features = calculate_trajectory_features(trajectory)
                        self.X.append(features)
                        self.y.append(CLASS_MAP[ship_type])
                        self.filenames.append(os.path.basename(csv_file))
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def get_data(self):
        return self.X, self.y, self.filenames

# -------------- Extra Trees 模型包装类 --------------
class ExtraTreesClassifierWrapper:
    def __init__(self, num_classes=4, params=None):
        self.num_classes = num_classes
        self.models = []
        self.params = params or self.get_default_params()
    
    def get_default_params(self):
        """获取默认的Extra Trees参数"""
        return {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 0.6,
            'bootstrap': True,
            'max_samples': 0.8,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def train(self, X_train, y_train, epoch_seed=42):
        """训练一个Extra Trees模型，使用不同的随机种子"""
        current_params = self.params.copy()
        current_params['random_state'] = epoch_seed
        model = ExtraTreesClassifier(**current_params)
        model.fit(X_train, y_train)
        return model
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.models:
            raise ValueError("No models available for prediction")
        
        # 对所有模型的预测进行平均（集成预测）
        all_preds = []
        for model in self.models:
            preds = model.predict_proba(X)
            all_preds.append(preds)
        
        return np.mean(all_preds, axis=0)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# -------------- 早停机制 --------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improvement = score - self.best_score
        else:
            improvement = self.best_score - score
            
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

# -------------- TOP-K 模型保存 --------------
class TopkSaver:
    def __init__(self, k=5, save_dir=MODEL_DIR):
        self.k = k
        self.heap = []   # (f1_score, epoch, path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def push(self, f1_val, epoch, model, feature_names=None):
        path = os.path.join(self.save_dir, f'epoch_{epoch}_f1_{f1_val:.4f}.pkl')
        # 保存模型和特征名称
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'f1_score': f1_val,
            'epoch': epoch
        }
        joblib.dump(model_data, path)
        heapq.heappush(self.heap, (f1_val, epoch, path))
        if len(self.heap) > self.k:
            _, _, old_path = heapq.heappop(self.heap)
            if os.path.exists(old_path):
                os.remove(old_path)

    def best_checkpoints(self):
        return sorted(self.heap, key=lambda x: -x[0])

# -------------- 训练和验证函数 --------------
def train_and_validate(model_wrapper, X_train, y_train, X_val, y_val, epoch, total_epochs):
    """训练和验证一个随机森林模型"""
    # 创建进度条
    pbar = tqdm(total=1, desc=f'Epoch {epoch+1}/{total_epochs} [RF Train+Val]', 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    # 训练一个新的模型，使用 epoch+seed 作为随机状态
    rf_model = model_wrapper.train(X_train, y_train, epoch_seed=42 + epoch)
    model_wrapper.models = [rf_model]  # 在当前 epoch 只保留这一个模型进行评价
    
    # 验证
    val_preds_proba = rf_model.predict_proba(X_val)
    val_preds = np.argmax(val_preds_proba, axis=1)
    
    # 计算指标
    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average='weighted', zero_division=0)
    recall = recall_score(y_val, val_preds, average='macro', zero_division=0)
    f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    
    pbar.update(1)
    pbar.close()
    
    return 0, accuracy, accuracy, precision, recall, f1, val_preds, y_val

def evaluate_model(model_wrapper, X, y, epoch, total_epochs, mode='Val'):
    """评估模型集成"""
    # 创建进度条
    pbar = tqdm(total=1, desc=f'Epoch {epoch+1}/{total_epochs} [{mode}]', 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    # 预测
    preds_proba = model_wrapper.predict_proba(X)
    preds = np.argmax(preds_proba, axis=1)
    
    # 计算指标
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, average='weighted', zero_division=0)
    recall = recall_score(y, preds, average='macro', zero_division=0)
    f1 = f1_score(y, preds, average='weighted', zero_division=0)
    
    pbar.update(1)
    pbar.close()
    
    return 0, accuracy, precision, recall, f1, preds, y

# -------------- 测试和集成函数 --------------
def test_model(rf_model, X_test, y_test, filenames, model_name=""):
    """测试单个随机森林模型"""
    # 创建进度条
    pbar = tqdm(total=1, desc=f"Testing {model_name}", 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    # 预测
    preds_proba = rf_model.predict_proba(X_test)
    preds = np.argmax(preds_proba, axis=1)
    
    # 计算指标
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    f1_val = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'traj_no': filenames,
        'test_shiptype': [CLASS_NAMES[p] for p in preds],
        'real_shiptype': [CLASS_NAMES[l] for l in y_test]
    })
    
    pbar.update(1)
    pbar.close()
    
    return accuracy, precision, recall, f1_val, results_df

def integrate_results(test_results_list, weights):
    """集成多个模型的测试结果"""
    # 初始化投票计数器
    vote_counts = {}
    
    # 对每个轨迹进行加权投票
    for (f1_val, epoch, _), results_df in zip(weights, test_results_list):
        weight = f1_val  # 使用F1分数作为权重
        
        for _, row in results_df.iterrows():
            traj_no = row['traj_no']
            pred_type = row['test_shiptype']
            
            if traj_no not in vote_counts:
                vote_counts[traj_no] = {}
            
            if pred_type not in vote_counts[traj_no]:
                vote_counts[traj_no][pred_type] = 0
            
            vote_counts[traj_no][pred_type] += weight
    
    # 确定每个轨迹的最终预测类型
    integrated_results = []
    for traj_no, votes in vote_counts.items():
        # 找到票数最多的类型
        final_pred = max(votes.items(), key=lambda x: x[1])[0]
        integrated_results.append({
            'traj_no': traj_no,
            'test_shiptype': final_pred
        })
    
    return pd.DataFrame(integrated_results)

def integrate_by_shipno(integrated_df, test_dir):
    """按shipno进行轨迹集成"""
    # 提取base_shipno
    integrated_df['base_shipno'] = integrated_df['traj_no'].str.replace(r'_\d+\.csv$', '', regex=True)
    
    # 获取每个轨迹的长度
    trajectory_lengths = []
    for traj_no in integrated_df['traj_no']:
        found = False
        for ship_type in CLASS_NAMES:
            csv_path = os.path.join(test_dir, ship_type, traj_no)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    trajectory_lengths.append(len(df))
                    found = True
                    break
                except:
                    pass
        if not found:
            trajectory_lengths.append(1)
    
    integrated_df['length'] = trajectory_lengths
    
    # 按shipno进行加权投票
    def resolve_group(df):
        votes = {}
        for _, row in df.iterrows():
            pred_type = row['test_shiptype']
            weight = row['length']
            
            if pred_type not in votes:
                votes[pred_type] = 0
            votes[pred_type] += weight
        
        # 返回票数最多的类型
        return max(votes.items(), key=lambda x: x[1])[0]
    
    # 获取真实标签
    def get_real_shiptype(base_shipno):
        for ship_type in CLASS_NAMES:
            pattern = os.path.join(test_dir, ship_type, f"{base_shipno}_*.csv")
            if glob.glob(pattern):
                return ship_type
        return "Unknown"
    
    final_df = integrated_df.groupby('base_shipno').apply(resolve_group).reset_index(name='test_shiptype')
    final_df['real_shiptype'] = final_df['base_shipno'].apply(get_real_shiptype)
    final_df = final_df.rename(columns={'base_shipno': 'shipno'})
    
    return final_df

# -------------- 特征名称 --------------
def get_feature_names():
    """获取特征名称"""
    return [
        # 距离相关特征
        'total_distance', 'mean_segment_distance', 'std_segment_distance', 
        'max_segment_distance', 'min_segment_distance',
        # 速度相关特征
        'mean_sog', 'std_sog', 'max_sog', 'min_sog', 'median_sog', 'high_speed_ratio',
        # 航向相关特征
        'mean_cog', 'std_cog', 'sharp_turn_ratio',
        # 高度变化特征
        'mean_delta_h', 'std_delta_h', 'max_delta_h', 'min_delta_h', 'total_abs_delta_h',
        # 时间相关特征
        'duration', 'mean_time_interval', 'std_time_interval', 'sampling_frequency',
        # 轨迹形状特征
        'start_end_distance', 'efficiency', 'mean_direction_change', 'std_direction_change',
        # 统计特征
        'num_points', 'sog_25p', 'sog_75p', 'cog_25p', 'cog_75p'
    ]

# -------------- 主流程 --------------
def main():
    # 初始化日志文件
    with open(LOG_FILE, 'w') as f:
        f.write("Extra Trees Ablation Study Training Log\n")
        f.write(f"Start Time: {datetime.now()}\n")
        f.write("="*50 + "\n")
    
    def log_message(message):
        print(message)
        with open(LOG_FILE, 'a') as f:
            f.write(message + "\n")
    
    # 1. 加载数据并提取特征
    log_message("Loading datasets and extracting features...")
    
    log_message("Processing training data...")
    train_dataset = ShipTrajectoryFeatureDataset(TRAIN_DIR)
    X_train, y_train, train_filenames = train_dataset.get_data()
    
    log_message("Processing validation data...")
    val_dataset = ShipTrajectoryFeatureDataset(VAL_DIR)
    X_val, y_val, val_filenames = val_dataset.get_data()
    
    log_message("Processing test data...")
    test_dataset = ShipTrajectoryFeatureDataset(TEST_DIR)
    X_test, y_test, test_filenames = test_dataset.get_data()
    
    log_message(f"Train samples: {len(X_train)}")
    log_message(f"Val samples: {len(X_val)}")
    log_message(f"Test samples: {len(X_test)}")
    log_message(f"Feature dimension: {X_train.shape[1]}")
    
    # 保存特征名称
    feature_names = get_feature_names()
    feature_info_df = pd.DataFrame({
        'feature_index': range(len(feature_names)),
        'feature_name': feature_names
    })
    feature_info_df.to_csv(os.path.join(RUN_DIR, "feature_names.csv"), index=False)
    
    # 2. 初始化模型包装器
    model_wrapper = ExtraTreesClassifierWrapper(num_classes=len(CLASS_NAMES))
    
    saver = TopkSaver(k=5)
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    # 3. 训练循环
    log_message("\nStarting training loop...")
    best_f1 = 0.0
    total_epochs = 10  # 随机森林通常不需要很多“epoch”，这里我们训练多个并集成
    
    for epoch in range(total_epochs):
        # 训练和验证
        tr_loss, tr_acc, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = train_and_validate(
            model_wrapper, X_train, y_train, X_val, y_val, epoch, total_epochs)
        
        # 显示epoch结果
        print("\n" + "="*60)
        print(f"Epoch {epoch+1:02d}/{total_epochs} - RESULTS:")
        print(f"  Train Acc: {tr_acc:.4f} (approx)")
        print(f"  Val   - Acc: {val_acc:.4f}")
        print(f"          Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print("="*60 + "\n")
        
        # 记录日志
        log_message(f"Epoch {epoch+1:02d}:")
        log_message(f"  Val - Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
        
        # 保存top-k模型（这里保存的是当前epoch训练出的单棵森林）
        saver.push(val_f1, epoch+1, model_wrapper.models[0], feature_names)
        
        # 早停检查
        if early_stopping(val_f1):
            log_message(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # 4. 使用TOP-K模型进行测试
    log_message("\n" + "="*50)
    log_message("Testing with top-5 models...")
    
    ckpts = saver.best_checkpoints()
    log_message(f"Using top {len(ckpts)} models for testing:")
    
    test_results = []
    test_metrics = []
    
    for i, (f1_val, epoch, path) in enumerate(ckpts):
        print(f"\nTesting Model {i+1}: Epoch {epoch}, Val F1: {f1_val:.4f}")
        log_message(f"Model {i+1}: Epoch {epoch}, Val F1: {f1_val:.4f}")
        
        # 加载模型
        model_data = joblib.load(path)
        rf_model = model_data['model']
        
        # 测试
        accuracy, precision, recall, test_f1, results_df = test_model(
            rf_model, X_test, y_test, test_filenames, f"Model_{epoch}")
        
        # 显示测试结果
        print(f"  Test Accuracy: {accuracy:.4f}, F1: {test_f1:.4f}")
        
        # 保存单个模型的测试结果
        test_csv_path = os.path.join(RUN_DIR, f"test_results_epoch_{epoch}.csv")
        results_df.to_csv(test_csv_path, index=False, encoding='utf-8')
        
        # 记录指标
        test_metrics.append({
            'epoch': epoch,
            'val_f1': f1_val,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': test_f1
        })
        
        test_results.append(results_df)
        log_message(f"  Test Results - Acc: {accuracy:.4f}, F1: {test_f1:.4f}")
    
    # 保存测试指标汇总
    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df.to_csv(os.path.join(RUN_DIR, "test_metrics_summary.csv"), index=False)
    
    # 5. 加权投票集成
    log_message("\n" + "="*50)
    log_message("Performing weighted voting integration...")
    
    # 第一步：模型集成
    integrated_df = integrate_results(test_results, ckpts)
    
    # 第二步：按shipno集成
    final_integrated_df = integrate_by_shipno(integrated_df, TEST_DIR)
    
    # 保存最终集成结果
    integrated_csv_path = os.path.join(RUN_DIR, "integrated_result.csv")
    final_integrated_df.to_csv(integrated_csv_path, index=False, encoding='utf-8')
    
    # 6. 计算最终集成结果的指标
    integrated_accuracy = accuracy_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'])
    integrated_precision = precision_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'], 
                                         average='weighted', zero_division=0)
    integrated_recall = recall_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'], 
                                   average='macro', zero_division=0)
    integrated_f1 = f1_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'], 
                            average='weighted', zero_division=0)
    
    log_message("\nFINAL INTEGRATED RESULTS:")
    log_message(f"Accuracy: {integrated_accuracy:.4f}")
    log_message(f"Precision: {integrated_precision:.4f}")
    log_message(f"Recall: {integrated_recall:.4f}")
    log_message(f"F1 Score: {integrated_f1:.4f}")
    
    # 保存最终指标
    final_metrics = {
        'integrated_accuracy': integrated_accuracy,
        'integrated_precision': integrated_precision,
        'integrated_recall': integrated_recall,
        'integrated_f1': integrated_f1
    }
    pd.DataFrame([final_metrics]).to_csv(os.path.join(RUN_DIR, "final_integrated_metrics.csv"), index=False)
    
    # 保存特征重要性（使用最佳 epoch 的模型）
    if ckpts:
        best_model_data = joblib.load(ckpts[0][2])
        best_rf = best_model_data['model']
        feature_importance = best_rf.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(RUN_DIR, "feature_importance.csv"), index=False)
    
    log_message(f"\nAll results saved to: {RUN_DIR}")
    print(f"\nTraining completed. Results in: {RUN_DIR}")

if __name__ == '__main__':
    main()
