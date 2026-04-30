import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import datetime
import json

# ==================== 配置类 ====================
class Config:
    # 路径配置
    BASE_DIR = "G:/drug-ML/circle-detect-96"
    IMAGE_DIR = os.path.join(BASE_DIR, "DATA_96/circle")
    LABEL_DIR = os.path.join(BASE_DIR, "DATA_96/loc")
    MODEL_SAVE_DIR = "models"
    RESULTS_DIR = "results"
    
    # 创建必要的目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 500
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 0
    
    # 模型参数
    IMAGE_SIZE = 256
    NUM_KEYPOINTS = 96
    
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据划分
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 实验名称（用于保存结果）
    EXPERIMENT_NAME = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ==================== 自定义JSON编码器 ====================
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super(NumpyEncoder, self).default(obj)

# ==================== 数据集类 ====================
class CircleDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=256, mode='train'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.mode = mode
        
        # 获取所有图片文件
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
        # 检查标签文件是否存在
        self.valid_files = []
        for img_path in self.image_files:
            label_path = self.label_dir / f"{img_path.stem}.npz"
            if label_path.exists():
                self.valid_files.append(img_path)
        
        self.image_files = self.valid_files
        
        # 划分数据集
        n_total = len(self.image_files)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        if mode == 'train':
            self.image_files = self.image_files[:n_train]
        elif mode == 'val':
            self.image_files = self.image_files[n_train:n_train+n_val]
        else:  # test
            self.image_files = self.image_files[n_train+n_val:]
        
        print(f"{mode} dataset: {len(self.image_files)} images")
        
        # 数据增强配置
        self.transform = self.get_transform(mode)
    
    def get_transform(self, mode):
        if mode == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # 加载图片
            img_path = self.image_files[idx]
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 记录原始尺寸
            h, w = image.shape[:2]
            
            # 加载关键点
            label_path = self.label_dir / f"{img_path.stem}.npz"
            data = np.load(str(label_path))
            
            # 处理关键点数据
            if 'x' in data and 'y' in data:
                x_coords = data['x']  
                y_coords = data['y']  #
                
                # 展平并合并
                x_flat = x_coords.flatten().astype(np.float32)
                y_flat = y_coords.flatten().astype(np.float32)
                
                # 合并为(48, 2)数组
                keypoints = np.stack([x_flat, y_flat], axis=1)
                
            else:
                raise ValueError(f"NPZ file missing x or y keys: {label_path}")
            
            # 归一化关键点坐标到[0, 1]
            keypoints_normalized = keypoints.copy().astype(np.float32)
            keypoints_normalized[:, 0] = keypoints[:, 0] / w
            keypoints_normalized[:, 1] = keypoints[:, 1] / h
            
            # 展平为96维向量
            keypoints_flat = keypoints_normalized.flatten()
            
            # 应用数据增强
            if self.transform:
                transformed = self.transform(image=image)
                image_tensor = transformed['image']
            else:
                image_resized = cv2.resize(image, (self.image_size, self.image_size))
                image_tensor = torch.from_numpy(image_resized).float()
                image_tensor = image_tensor.permute(2, 0, 1) / 255.0
            
            return {
                'image': image_tensor,
                'keypoints': torch.FloatTensor(keypoints_flat),
                'original_size': torch.FloatTensor([h, w]),
                'image_path': str(img_path)
            }
            
        except Exception as e:
            print(f"Error processing image {self.image_files[idx]}: {str(e)}")
            raise e

# ==================== 模型定义 ====================
class KeypointNet(nn.Module):
    def __init__(self, num_keypoints=96):
        super(KeypointNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # 计算特征图大小：256x256 -> 16x16
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_keypoints * 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        
        # 重塑为(batch_size, num_keypoints, 2)
        x = x.view(-1, self.fc3.out_features // 2, 2)
        
        return x

# ==================== 训练和验证函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    batch_losses = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        keypoints = batch['keypoints'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred_keypoints = model(images)
        
        # 计算损失
        gt_keypoints = keypoints.view(pred_keypoints.shape)
        loss = criterion(pred_keypoints, gt_keypoints)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_losses.append(loss.item())
    
    return running_loss / len(dataloader), batch_losses

def validate_epoch(model, dataloader, criterion, device, calculate_metrics=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    distances = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            original_sizes = batch['original_size'].cpu().numpy()
            
            # 预测
            pred_keypoints = model(images)
            
            # 计算损失
            gt_keypoints = keypoints.view(pred_keypoints.shape)
            loss = criterion(pred_keypoints, gt_keypoints)
            running_loss += loss.item()
            
            if calculate_metrics:
                # 将预测的关键点转换回原始尺寸
                batch_size = pred_keypoints.size(0)
                for i in range(batch_size):
                    orig_h, orig_w = original_sizes[i]
                    
                    # 获取预测和真实关键点
                    pred_kp = pred_keypoints[i].cpu().numpy()
                    gt_kp = gt_keypoints[i].cpu().numpy()
                    
                    # 反归一化到原始尺寸
                    pred_kp[:, 0] = pred_kp[:, 0] * orig_w
                    pred_kp[:, 1] = pred_kp[:, 1] * orig_h
                    gt_kp[:, 0] = gt_kp[:, 0] * orig_w
                    gt_kp[:, 1] = gt_kp[:, 1] * orig_h
                    
                    # 计算每个关键点的欧氏距离
                    dist = np.sqrt(np.sum((pred_kp - gt_kp) ** 2, axis=1))
                    distances.extend(dist)
                    
                    all_preds.append(pred_kp)
                    all_targets.append(gt_kp)
    
    avg_loss = running_loss / len(dataloader)
    
    if calculate_metrics and distances:
        distances = np.array(distances)
        metrics = {
            'avg_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'p95_distance': float(np.percentile(distances, 95)),
            'max_distance': float(np.max(distances)),
            'min_distance': float(np.min(distances))
        }
        return avg_loss, metrics, all_preds, all_targets
    else:
        return avg_loss, None, None, None

# ==================== 可视化函数 ====================
def plot_training_history(train_losses, val_losses, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制损失对数曲线
    axes[1].semilogy(train_losses, label='Train Loss', linewidth=2)
    axes[1].semilogy(val_losses, label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Training and Validation Loss (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_distribution(distances, save_path):
    """绘制误差分布"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 直方图
    axes[0].hist(distances, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Error Distance (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Keypoint Error Distribution')
    axes[0].axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    axes[0].axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[1].boxplot(distances, vert=True, patch_artist=True)
    axes[1].set_ylabel('Error Distance (pixels)')
    axes[1].set_title('Keypoint Error Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Mean: {np.mean(distances):.2f}\nMedian: {np.median(distances):.2f}\nStd: {np.std(distances):.2f}\n95th percentile: {np.percentile(distances, 95):.2f}'
    axes[1].text(1.2, np.percentile(distances, 95), stats_text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, dataset, device, num_samples=5, save_dir='results/visualizations'):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        batch = dataset[sample_idx]
        
        image = batch['image'].unsqueeze(0).to(device)
        gt_keypoints = batch['keypoints']
        orig_h, orig_w = batch['original_size'].cpu().numpy()
        
        with torch.no_grad():
            pred_keypoints = model(image)
        
        # 转换到CPU和numpy
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # 获取预测关键点
        pred_kp = pred_keypoints.squeeze().cpu().numpy()
        pred_kp[:, 0] = pred_kp[:, 0] * orig_w
        pred_kp[:, 1] = pred_kp[:, 1] * orig_h
        
        # 获取真实关键点
        gt_kp = gt_keypoints.view(-1, 2).cpu().numpy()
        gt_kp[:, 0] = gt_kp[:, 0] * orig_w
        gt_kp[:, 1] = gt_kp[:, 1] * orig_h
        
        # 原始图像
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # 真实关键点
        axes[idx, 1].imshow(image_np)
        for x, y in gt_kp:
            axes[idx, 1].scatter(x * (image_np.shape[1] / orig_w), 
                               y * (image_np.shape[0] / orig_h), 
                               c='green', s=20, marker='o', label='Ground Truth')
        axes[idx, 1].set_title(f'Ground Truth ({len(gt_kp)} points)')
        axes[idx, 1].axis('off')
        
        # 预测关键点
        axes[idx, 2].imshow(image_np)
        for x, y in pred_kp:
            axes[idx, 2].scatter(x * (image_np.shape[1] / orig_w), 
                               y * (image_np.shape[0] / orig_h), 
                               c='red', s=20, marker='x', label='Prediction')
        
        # 绘制误差线
        for i in range(min(len(gt_kp), len(pred_kp))):
            if i < len(gt_kp) and i < len(pred_kp):
                axes[idx, 2].plot([gt_kp[i, 0] * (image_np.shape[1] / orig_w), 
                                 pred_kp[i, 0] * (image_np.shape[1] / orig_w)], 
                                [gt_kp[i, 1] * (image_np.shape[0] / orig_h), 
                                 pred_kp[i, 1] * (image_np.shape[0] / orig_h)], 
                                'y-', alpha=0.5, linewidth=1)
        
        axes[idx, 2].set_title(f'Predictions ({len(pred_kp)} points)')
        axes[idx, 2].legend()
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(save_dir, f'predictions_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def save_training_summary(config, train_losses, val_losses, metrics, save_dir):
    """保存训练总结"""
    # 确保所有数据都是Python内置类型
    config_dict = config.__dict__.copy()
    
    summary = {
        'experiment_name': config.EXPERIMENT_NAME,
        'config': config_dict,
        'training_stats': {
            'final_train_loss': float(train_losses[-1]) if train_losses else None,
            'final_val_loss': float(val_losses[-1]) if val_losses else None,
            'best_val_loss': float(min(val_losses)) if val_losses else None,
            'best_epoch': int(val_losses.index(min(val_losses))) if val_losses else None,
            'num_epochs': int(len(train_losses))
        },
        'validation_metrics': metrics
    }
    
    # 保存为JSON
    summary_path = os.path.join(save_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # 保存为CSV
    if train_losses and val_losses:
        df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': [float(loss) for loss in train_losses],
            'val_loss': [float(loss) for loss in val_losses]
        })
        csv_path = os.path.join(save_dir, 'loss_history.csv')
        df.to_csv(csv_path, index=False)
    
    return summary

# ==================== 主训练函数 ====================
def main():
    cfg = Config()
    
    # 创建实验目录
    exp_dir = os.path.join(cfg.RESULTS_DIR, cfg.EXPERIMENT_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print("=" * 60)
    
    print(f"Device: {cfg.DEVICE}")
    print(f"Image directory: {cfg.IMAGE_DIR}")
    print(f"Label directory: {cfg.LABEL_DIR}")
    print(f"Results saved to: {exp_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(cfg.IMAGE_DIR):
        print(f"Error: Image directory does not exist: {cfg.IMAGE_DIR}")
        return
    
    if not os.path.exists(cfg.LABEL_DIR):
        print(f"Error: Label directory does not exist: {cfg.LABEL_DIR}")
        return
    
    # 创建数据集
    print(f"\nCreating datasets...")
    train_dataset = CircleDataset(
        cfg.IMAGE_DIR, 
        cfg.LABEL_DIR, 
        image_size=cfg.IMAGE_SIZE,
        mode='train'
    )
    
    val_dataset = CircleDataset(
        cfg.IMAGE_DIR,
        cfg.LABEL_DIR,
        image_size=cfg.IMAGE_SIZE,
        mode='val'
    )
    
    test_dataset = CircleDataset(
        cfg.IMAGE_DIR,
        cfg.LABEL_DIR,
        image_size=cfg.IMAGE_SIZE,
        mode='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    # 初始化模型
    print(f"\nInitializing model...")
    model = KeypointNet(num_keypoints=cfg.NUM_KEYPOINTS)
    model = model.to(cfg.DEVICE)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {cfg.EPOCHS} epochs")
    print("-" * 60)
    
    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        
        # 训练
        train_loss, batch_losses = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        train_losses.append(float(train_loss))
        
        # 验证
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss, metrics, _, _ = validate_epoch(model, val_loader, criterion, cfg.DEVICE, calculate_metrics=True)
            print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            
            if metrics:
                print(f"Validation metrics:")
                print(f"  Average error: {metrics['avg_distance']:.2f} pixels")
                print(f"  Median error: {metrics['median_distance']:.2f} pixels")
                print(f"  95th percentile: {metrics['p95_distance']:.2f} pixels")
        else:
            val_loss, _, _, _ = validate_epoch(model, val_loader, criterion, cfg.DEVICE, calculate_metrics=False)
            print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        
        val_losses.append(float(val_loss))
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(val_loss),
                'config': cfg.__dict__
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(val_loss),
                'config': cfg.__dict__
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # 每10个epoch可视化一次预测
            viz_path = visualize_predictions(model, val_dataset, cfg.DEVICE, num_samples=3, 
                                           save_dir=os.path.join(exp_dir, 'visualizations'))
            print(f"Prediction visualization saved to: {viz_path}")
    
    # 最终验证（在测试集上）
    print(f"\n" + "=" * 60)
    print("Final test evaluation...")
    print("=" * 60)
    
    test_loss, test_metrics, all_preds, all_targets = validate_epoch(
        model, test_loader, criterion, cfg.DEVICE, calculate_metrics=True
    )
    
    print(f"Test Loss: {test_loss:.6f}")
    if test_metrics:
        print(f"Test metrics:")
        print(f"  Average error: {test_metrics['avg_distance']:.2f} pixels")
        print(f"  Median error: {test_metrics['median_distance']:.2f} pixels")
        print(f"  Standard deviation: {test_metrics['std_distance']:.2f} pixels")
        print(f"  95th percentile: {test_metrics['p95_distance']:.2f} pixels")
        print(f"  Maximum error: {test_metrics['max_distance']:.2f} pixels")
        print(f"  Minimum error: {test_metrics['min_distance']:.2f} pixels")
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save({
        'epoch': cfg.EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(test_loss),
        'test_metrics': test_metrics,
        'config': cfg.__dict__
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # 可视化训练历史
    loss_plot_path = os.path.join(exp_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, loss_plot_path)
    print(f"Training history plot saved to: {loss_plot_path}")
    
    # 可视化误差分布
    if test_metrics and all_preds:
        # 计算所有距离
        all_distances = []
        for pred, target in zip(all_preds, all_targets):
            dist = np.sqrt(np.sum((pred - target) ** 2, axis=1))
            all_distances.extend(dist)
        
        error_plot_path = os.path.join(exp_dir, 'error_distribution.png')
        plot_error_distribution(np.array(all_distances), error_plot_path)
        print(f"Error distribution plot saved to: {error_plot_path}")
    
    # 保存训练总结
    summary = save_training_summary(cfg, train_losses, val_losses, test_metrics, exp_dir)
    summary_path = os.path.join(exp_dir, 'training_summary.json')
    print(f"Training summary saved to: {summary_path}")
    
    # 打印最终总结
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Experiment name: {cfg.EXPERIMENT_NAME}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    if test_metrics:
        print(f"Test set average error: {test_metrics['avg_distance']:.2f} pixels")
    print(f"All results saved to: {exp_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()