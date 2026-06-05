import argparse
import os
import torch
from exp.exp_ship_classification1 import Exp_Ship_Classification
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ship Classification')
    parser.add_argument('--resume', action='store_true', help='是否从断点恢复训练')
    parser.add_argument('--resume_epoch', type=int, default=0, help='从指定epoch恢复训练')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--pooling', type=str, default='attention', help='注意力池化')
    parser.add_argument('--min_seq_len', type=int, default=300, help='input trajectory sequence length')


    # RANDOM SEED
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # BASIC CONFIG
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='ship_cls', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ShipClassifier',
                        help='model name, options: [ShipClassifier, TransformerClassifier, etc.]')
    parser.add_argument('--task_name', type=str, required=False, default='ship_classification', help='task name')

    # DATALOADER
    parser.add_argument('--data', type=str, required=True, default='Ship', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ship/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ship_data.csv', help='ship trajectory data file')
    parser.add_argument('--label_path', type=str, default='ship_labels.csv', help='ship classification labels file')
    parser.add_argument('--features', type=str, default='trajectory',
                        help='input features type, options:[trajectory, static, combined]')
    parser.add_argument('--target', type=str, default='ship_type', help='target classification label')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # SHIP TRAJECTORY CONFIG
    parser.add_argument('--seq_len', type=int, default=128, help='input trajectory sequence length')
    parser.add_argument('--n_features', type=int, default=4, help='number of features (lat, lon, sog, cog)')
    parser.add_argument('--time_window', type=int, default=3600, help='time window for trajectory (seconds)')
    parser.add_argument('--min_points', type=int, default=10, help='minimum trajectory points required')
    parser.add_argument('--max_gap', type=int, default=300, help='maximum gap between trajectory points (seconds)')

    # SHIP CLASSIFICATION SPECIFIC
    parser.add_argument('--num_classes', type=int, default=7, help='number of ship types to classify')
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['Cargo', 'Tanker', 'Fishing', 'Passenger', 'Tug', 'Military', 'Other'],
                        help='ship class names')
    parser.add_argument('--use_ship_static', type=bool, default=False,
                        help='use static ship features (length, width, etc.)')
    parser.add_argument('--static_features', type=int, default=0, help='number of static features if used')

    # MODEL CONFIG
    parser.add_argument('--d_model', type=int, default=256, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='feed forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--embed_type', type=str, default='learned',
                        help='embedding type, options:[learned, positional]')

    # DATA AUGMENTATION
    parser.add_argument('--use_augmentation', type=bool, default=True, help='use data augmentation')
    parser.add_argument('--noise_level', type=float, default=0.01, help='noise level for augmentation')
    parser.add_argument('--rotation_angle', type=float, default=10.0, help='rotation angle for augmentation (degrees)')
    parser.add_argument('--time_warp', type=bool, default=True, help='use time warping augmentation')

    # TRAINING CONFIG
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='test data ratio')
    parser.add_argument('--stratify', type=bool, default=True, help='stratified split by ship type')

    # OPTIMIZATION
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer type')
    parser.add_argument('--scheduler', type=str, default='onecycle', help='learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')

    # LOSS AND REGULARIZATION
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='loss function type')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')
    parser.add_argument('--focal_loss', type=bool, default=False, help='use focal loss for imbalanced data')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
    parser.add_argument('--class_weights', type=bool, default=True, help='use class weights for imbalanced data')

    # EVALUATION
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        help='evaluation metrics')
    parser.add_argument('--save_predictions', type=bool, default=True, help='save test predictions')
    parser.add_argument('--plot_results', type=bool, default=True, help='plot confusion matrix and training curves')

    # LEARNING RATE SCHEDULING
    parser.add_argument('--lradj', type=str, default='onecycle', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='percentage of cycle spent increasing lr')
    parser.add_argument('--max_lr_factor', type=float, default=10.0, help='maximum lr factor for onecycle')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # MODEL SPECIFIC PARAMETERS
    parser.add_argument('--model_type', type=str, default='transformer',
                        help='model architecture type, options:[transformer, lstm, gru, cnn, hybrid]')
    parser.add_argument('--use_positional_encoding', type=bool, default=True, help='use positional encoding')
    parser.add_argument('--max_seq_len', type=int, default=512, help='maximum sequence length for positional encoding')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # LOGGING AND SAVING
    parser.add_argument('--save_best_only', type=bool, default=True, help='save only the best model')
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval during training')
    parser.add_argument('--save_interval', type=int, default=10, help='model saving interval (epochs)')
    parser.add_argument('--exp_description', type=str, default='ship_classification_experiment',
                        help='experiment description')

    # RESUME TRAINING
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')

    # INFERENCE
    parser.add_argument('--do_predict', action='store_true', help='whether to predict on new data')
    parser.add_argument('--predict_data_path', type=str, default='', help='path to data for prediction')

    args = parser.parse_args()

    # Set random seed for reproducibility
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set task name based on data path
    args.task_name = args.data_path.split('.')[0] if '.' in args.data_path else args.data_path

    # GPU configuration
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Validate data split ratios
    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        print("Warning: train_ratio + val_ratio + test_ratio != 1.0, normalizing...")
        total = args.train_ratio + args.val_ratio + args.test_ratio
        args.train_ratio /= total
        args.val_ratio /= total
        args.test_ratio /= total

    print('Args in experiment:')
    print(args)

    # Set experiment class
    Exp = Exp_Ship_Classification

    if args.is_training:
        for ii in range(args.itr):
            # Setting record of experiments
            setting = '{}_{}_{}_sl{}_dm{}_nl{}_nh{}_dr{}_bs{}_lr{}_ep{}_{}'.format(
                args.model_id,
                args.model_type,
                args.task_name,
                args.seq_len,
                args.d_model,
                args.n_layers,
                args.n_heads,
                args.dropout,
                args.batch_size,
                args.learning_rate,
                args.train_epochs,
                ii)

            exp = Exp(args)  # Initialize experiment

            print('>>>>>>>Start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>Testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_results = exp.test(setting)

            # Print final results for this iteration
            print(f">>>>>>>Iteration {ii + 1} Results:")
            print(f"Test Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
            print(f"Test Micro F1: {test_results['micro_f1']:.4f}")

            if args.do_predict and args.predict_data_path:
                print('>>>>>>>Predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict_new_data(args.predict_data_path, setting)

            torch.cuda.empty_cache()

    else:
        # Testing only mode
        ii = 0
        setting = '{}_{}_{}_sl{}_dm{}_nl{}_nh{}_dr{}_bs{}_lr{}_ep{}_{}'.format(
            args.model_id,
            args.model_type,
            args.task_name,
            args.seq_len,
            args.d_model,
            args.n_layers,
            args.n_heads,
            args.dropout,
            args.batch_size,
            args.learning_rate,
            args.train_epochs,
            ii)

        exp = Exp(args)  # Initialize experiment
        print('>>>>>>>Testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_results = exp.test(setting, test=1)

        print(f">>>>>>>Final Test Results:")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
        print(f"Test Micro F1: {test_results['micro_f1']:.4f}")

        torch.cuda.empty_cache()

# 使用示例：
# 训练模型:
# python run_ship_classification.py --is_training 1 --model_id ship_cls_v1 --model ShipClassifier --data Ship --data_path ship_trajectories.csv --seq_len 128 --batch_size 32 --train_epochs 100

# 仅测试:
# python run_ship_classification.py --is_training 0 --model_id ship_cls_v1 --model ShipClassifier --data Ship --data_path ship_trajectories.csv

# 使用GPU训练:
# python run_ship_classification.py --is_training 1 --model_id ship_cls_v1 --model ShipClassifier --data Ship --use_gpu True --gpu 0

# 多GPU训练:
# python run_ship_classification.py --is_training 1 --model_id ship_cls_v1 --model ShipClassifier --data Ship --use_multi_gpu --devices 0,1,2,3