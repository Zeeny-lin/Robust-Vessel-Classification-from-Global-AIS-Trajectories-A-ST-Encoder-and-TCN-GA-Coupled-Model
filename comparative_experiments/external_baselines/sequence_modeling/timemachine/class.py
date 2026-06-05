from exp.exp_basic import Exp_Basic
from data_provider.ship_data_loader import ship_data_provider
from models.TimeMachine import ShipClassificationModel
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import pandas as pd
import os
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from main.SpatialRelationEncoder import GridCellSpatialRelationLocationEncoder

warnings.filterwarnings('ignore')


class Exp_Ship_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Ship_Classification, self).__init__(args)
        self.geo_encoder = GridCellSpatialRelationLocationEncoder(
            spa_embed_dim=64,
            coord_dim=2,
            frequency_num=8,
            max_radius=10000,
            min_radius=10,
            freq_init="geometric",
            device=self.device,  # 使用实际的设备
            ffn_act="relu",
            ffn_num_hidden_layers=1,
            ffn_dropout_rate=0.5,
            ffn_hidden_dim=256,
            ffn_use_layernormalize=True,
            ffn_skip_connection=True,
            ffn_context_str="GridCellSpatialRelationEncoder"
        )

    def _build_model(self):
        """构建分类模型"""
        # 设置模型配置
        self.args.n_features = 6  # lat, lon, sog, cog
        self.args.d_model = getattr(self.args, 'd_model', 256)
        self.args.n_layers = getattr(self.args, 'n_layers', 4)
        # num_classes 将在数据加载后设置

        model = ShipClassificationModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """获取数据"""
        data_set, data_loader = ship_data_provider(self.args, flag)

        # 设置类别数
        if not hasattr(self.args, 'num_classes'):
            self.args.num_classes = data_set.num_classes
            print(f"Set num_classes to {self.args.num_classes}")

            # 重新构建模型（如果需要）
            if hasattr(self, 'model'):
                self.model = self._build_model().to(self.device)

        return data_set, data_loader

    def _select_optimizer(self):
        """选择优化器"""
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 1e-4)
        )
        return model_optim

    def _select_criterion(self):
        """选择损失函数"""
        # 可以根据数据分布选择不同的损失函数
        if getattr(self.args, 'label_smoothing', 0) > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def _process_geo_embedding(self, coords_input):
        """处理地理坐标编码，统一数据类型和格式"""
        if coords_input is None:
            return None

        # 确保数据类型为numpy.ndarray
        if isinstance(coords_input, torch.Tensor):
            coords_input = coords_input.detach().cpu().numpy()
        elif isinstance(coords_input, list):
            coords_input = np.array(coords_input)

        # 确保数据格式正确 [batch_size, seq_len, 2]
        if coords_input.ndim == 2:
            # [seq_len, 2] -> [1, seq_len, 2]
            coords_input = np.expand_dims(coords_input, axis=0)
        elif coords_input.ndim == 4:
            # [batch_size, 1, seq_len, 2] -> [batch_size, seq_len, 2]
            coords_input = coords_input.squeeze(1)

        # 确保数据类型为float32
        coords_input = coords_input.astype(np.float32)

        try:
            # 使用geo_encoder进行编码
            geo_embedding = self.geo_encoder(coords_input)
            print(geo_embedding.type())

            # 转换为torch tensor并移到正确的设备
            if not isinstance(geo_embedding, torch.Tensor):
                geo_embedding = torch.tensor(geo_embedding, dtype=torch.float32)

            geo_embedding = geo_embedding.to(self.device)
            return geo_embedding

        except Exception as e:
            print(f"Error in geo encoding: {e}",geo_embedding.type())
            print(f"coords_input shape: {coords_input.shape}")
            print(f"coords_input dtype: {coords_input.dtype}")
            return None

    def vali(self, vali_data, vali_loader, criterion):
        """验证"""
        total_loss = []
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_filenames, coords_input) in enumerate(vali_loader):

                # 处理地理坐标编码
                geo_embedding = self._process_geo_embedding(coords_input)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().squeeze(-1).to(self.device)  # [batch_size]

                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, geo_embedding)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, geo_embedding)
                    loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())

                # 预测
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # 计算指标
        avg_loss = np.mean(total_loss)
        accuracy = accuracy_score(all_labels, all_preds)

        self.model.train()
        return avg_loss, accuracy

    def train(self, setting):
        """训练"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        resume_epoch = self.args.resume_epoch

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 学习率调度器
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=getattr(self.args, 'pct_start', 0.3),
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        # 训练历史
        train_losses = []
        val_losses = []
        val_accuracies = []
        if self.args.resume:
            checkpoint_path = os.path.join(path, f'checkpoint_{resume_epoch}.pth')
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.args.use_amp and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # 加载历史数据
            history_path = os.path.join(path, 'training_history.npy')
            if os.path.exists(history_path):
                history = np.load(history_path, allow_pickle=True).item()
                train_losses = history['train_loss'][:resume_epoch]
                val_losses = history['val_loss'][:resume_epoch]
                val_accuracies = history['val_accuracy'][:resume_epoch]

            print(f"Resuming training from epoch {resume_epoch + 1}")

        start_epoch = resume_epoch if self.args.resume else 0

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_filenames, coords_input) in enumerate(train_loader):
                # 处理地理坐标编码
                geo_embedding = self._process_geo_embedding(coords_input)

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().squeeze(-1).to(self.device)  # [batch_size]

                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, geo_embedding)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, geo_embedding)
                    loss = criterion(outputs, batch_y)

                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at epoch {epoch}, batch {i}, skipping...")
                    continue

                train_loss.append(loss.item())

                # 反向传播
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # 学习率调度
                if getattr(self.args, 'lradj', 'TST') == 'TST':
                    scheduler.step()

                # 打印进度
                if (i + 1) % 50 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")

            # 评估
            train_loss_avg = np.mean(train_loss) if train_loss else float('inf')
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss_avg:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Vali Acc: {vali_acc:.4f} "
                  f"Test Loss: {test_loss:.7f} Test Acc: {test_acc:.4f}")

            # 记录历史
            train_losses.append(train_loss_avg)
            val_losses.append(vali_loss)
            val_accuracies.append(vali_acc)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss_avg,
                'val_loss': vali_loss,
                'val_acc': vali_acc
            }
            if self.args.use_amp:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            # 保存模型
            checkpoint_path = os.path.join(path, f'checkpoint_{epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)

            # 早停检查
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学习率调整
            if getattr(self.args, 'lradj', 'TST') != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        # 保存训练历史
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        np.save(os.path.join(path, 'training_history.npy'), history)

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """测试"""
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        resume_epoch = self.args.resume_epoch

        if test:
            print('Loading model')
            if self.args.resume:
                checkpoint_path = os.path.join('./trainOut/' + setting, f'checkpoint_{resume_epoch}.pth')
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(
                    torch.load(os.path.join('./trainOut/' + setting, 'checkpoint.pth'))
                )
                print('checkpoint.pth')

        # 预测和真实标签
        all_preds = []
        all_labels = []
        all_probs = []
        all_filenames = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_filenames, coords_input) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().squeeze(-1).to(self.device)

                # 处理地理坐标编码
                geo_embedding = self._process_geo_embedding(coords_input)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, geo_embedding)
                else:
                    outputs = self.model(batch_x, geo_embedding)

                # 获取预测概率和预测标签
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_filenames.extend(batch_filenames)  # 存储文件名

        # 计算各种指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='micro', zero_division=0
        )

        # 打印结果
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}")
        print(f"Micro F1: {micro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}")

        # 获取类别名称
        if hasattr(train_data, 'get_label_names'):
            class_names = train_data.get_label_names().tolist()
        elif hasattr(train_data, 'label_encoder') and hasattr(train_data.label_encoder, 'classes_'):
            class_names = train_data.label_encoder.classes_.tolist()
        elif hasattr(train_data.dataset, 'get_label_names') if hasattr(train_data, 'dataset') else False:
            class_names = train_data.dataset.get_label_names().tolist()
        elif hasattr(train_data.dataset, 'label_encoder') if hasattr(train_data, 'dataset') else False:
            class_names = train_data.dataset.label_encoder.classes_.tolist()
        else:
            unique_labels = sorted(np.unique(all_labels))
            class_names = [str(i) for i in unique_labels]
            print(f"Warning: No class names found, using numeric labels: {class_names}")
            print("Available attributes in train_data:", [attr for attr in dir(train_data) if not attr.startswith('_')])

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 创建结果保存路径
        results_path = os.path.join('./results', setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # 处理概率数据
        prob_columns = {}
        all_probs = np.array(all_probs)
        num_classes = all_probs.shape[1]

        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            prob_columns[f'prob_{class_name}'] = all_probs[:, i]

        # 将索引转换为类别名称
        pred_class_names = [class_names[idx] if idx < len(class_names) else f"Class_{idx}" for idx in all_preds]
        true_class_names = [class_names[idx] if idx < len(class_names) else f"Class_{idx}" for idx in all_labels]

        # 创建主要结果DataFrame
        results_df = pd.DataFrame({
            'filename': all_filenames,
            'predictions': pred_class_names,
            'true_labels': true_class_names,
            'pred_index': all_preds,
            'true_index': all_labels,
            **prob_columns
        })

        # 保存主要结果
        results_df.to_csv(os.path.join(results_path, 'test_results.csv'), index=False)
        predictions_df = pd.DataFrame({
            'shipno': all_filenames,
            'shiptype': pred_class_names,
        })
        from datetime import datetime
        timestamp = datetime.now().strftime("%H%M%S")
        predictions_df.to_csv(os.path.join(results_path, f'predict_{timestamp}.csv'), index=False)

        # 保存评估指标到单独的CSV文件
        metrics_df = pd.DataFrame({
            'metric': ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
                       'micro_f1', 'micro_precision', 'micro_recall'],
            'value': [accuracy, macro_f1, macro_precision, macro_recall,
                      micro_f1, micro_precision, micro_recall]
        })
        metrics_df.to_csv(os.path.join(results_path, 'test_metrics.csv'), index=False)

        # 保存每个类别的详细指标
        class_metrics_df = pd.DataFrame({
            'class_name': class_names[:len(precision)],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        class_metrics_df.to_csv(os.path.join(results_path, 'class_metrics.csv'), index=False)

        # 保存混淆矩阵
        cm_df = pd.DataFrame(cm, index=class_names[:cm.shape[0]], columns=class_names[:cm.shape[1]])
        cm_df.to_csv(os.path.join(results_path, 'confusion_matrix.csv'))

        # 绘制并保存混淆矩阵
        self.plot_confusion_matrix(cm, class_names, results_path)

        # 如果有训练历史，绘制训练曲线
        history_path = os.path.join('./checkpoints', setting, 'training_history.npy')
        if os.path.exists(history_path):
            self.plot_training_curves(history_path, results_path)

        # 构建返回的结果字典
        test_results = {
            'filename': all_filenames,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'confusion_matrix': cm,
            'class_names': class_names
        }

        print(f"\nResults saved to: {results_path}")
        print("Files created:")
        print("- test_results.csv: 详细的预测结果")
        print("- test_metrics.csv: 总体评估指标")
        print("- class_metrics.csv: 各类别详细指标")
        print("- confusion_matrix.csv: 混淆矩阵")

        return test_results

    def plot_confusion_matrix(self, cm, class_names, save_path):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 创建标签（显示数量和百分比）
        labels = np.array([
            [f'{int(cm[i, j])}\n({cm_percent[i, j]:.1f}%)' for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        sns.heatmap(
            cm_percent,
            annot=labels,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage (%)'}
        )

        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {os.path.join(save_path, 'confusion_matrix.png')}")

    def plot_training_curves(self, history_path, save_path):
        """绘制训练曲线"""
        history = np.load(history_path, allow_pickle=True).item()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to {os.path.join(save_path, 'training_curves.png')}")

    def predict(self, data_loader):
        """对新数据进行预测"""
        all_preds = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, _, batch_filenames, coords_input in data_loader:
                batch_x = batch_x.float().to(self.device)

                # 处理地理坐标编码
                geo_embedding = self._process_geo_embedding(coords_input)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, geo_embedding)
                else:
                    outputs = self.model(batch_x, geo_embedding)

                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def save_model_info(self, setting):
        """保存模型信息"""
        info = {
            'model_type': 'ShipClassificationModel',
            'num_classes': self.args.num_classes,
            'n_features': self.args.n_features,
            'd_model': self.args.d_model,
            'n_layers': self.args.n_layers,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size,
            'train_epochs': self.args.train_epochs,
            'patience': self.args.patience
        }

        info_path = os.path.join('./trainOut', setting, 'model_info.npy')
        np.save(info_path, info)
        print(f"Model info saved to {info_path}")