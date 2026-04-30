import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 首先检查CUDA是否可用，如果不可用或有问题，强制使用CPU
try:
    if torch.cuda.is_available():
        # 测试CUDA是否正常工作
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        test_result = test_tensor * 2
        print("CUDA test passed, using GPU")
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
except Exception as e:
    print(f"CUDA initialization failed: {e}")
    print("Forcing CPU usage")
    device = torch.device('cpu')
    torch.cuda.is_available = lambda: False

print(f"Using device: {device}")

# 创建输出目录用于保存图表
output_dir = "model_output_plots"
os.makedirs(output_dir, exist_ok=True)

# 设置图形显示 - 配置为保存高质量PDF
plt.rcParams['pdf.fonttype'] = 42  # 确保文字在PDF中保持为可编辑的文字而不是曲线
plt.rcParams['ps.fonttype'] = 42   # 确保文字在PostScript中保持为可编辑的文字
plt.rcParams['font.family'] = 'Arial'  # 使用Arial字体，确保在AI中可编辑
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("Using PyTorch neural network for Cis and Trans prediction - RGB channels only")

# 1. 数据集创建函数 - 从CSV读取数据
def create_dataset_from_csv():
    """Create dataset from CSV file"""
    df = pd.read_csv('all_colors_data.csv')
    print(f"Successfully read data from CSV file, dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    
    if df.columns[-1] == 'Y1' and 'Y1' in df.columns:
        df = df.rename(columns={df.columns[-1]: 'Y2'})
        print("Renamed the last column to Y2")
    
    rgb_columns = ['A_B', 'A_G', 'A_R', 'B_B', 'B_G', 'B_R']
    
    missing_cols = [col for col in rgb_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Following RGB columns not found: {missing_cols}")
        rgb_columns = []
        for col in df.columns:
            if col.startswith('A_') and any(ch in col for ch in ['B', 'G', 'R']):
                rgb_columns.append(col)
            elif col.startswith('B_') and any(ch in col for ch in ['B', 'G', 'R']):
                rgb_columns.append(col)
    
    print(f"RGB feature columns ({len(rgb_columns)}): {rgb_columns}")
    print(f"Target columns: Cis (Y1), Trans (Y2)")
    
    X = df[rgb_columns].values.astype(np.float32)
    y = df[['Y1', 'Y2']].values.astype(np.float32)
    
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("Warning: Data contains NaN values, filling them")
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
    
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        print("Warning: Data contains Inf values, replacing them")
        X = np.where(np.isinf(X), 0, X)
        y = np.where(np.isinf(y), 0, y)
    
    print(f"\nData statistics:")
    print(f"RGB features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Cis (Y1) range: [{y[:, 0].min():.4f}, {y[:, 0].max():.4f}]")
    print(f"Trans (Y2) range: [{y[:, 1].min():.4f}, {y[:, 1].max():.4f}]")
    
    print(f"\nRGB feature statistics:")
    for i, col in enumerate(rgb_columns):
        print(f"{col}: mean={X[:, i].mean():.2f}, std={X[:, i].std():.2f}, "
              f"range=[{X[:, i].min():.0f}, {X[:, i].max():.0f}]")
    
    return X, y, rgb_columns, df

# 2. 数据预处理
def preprocess_data(X, y, test_size=0.2):
    """Data preprocessing function"""
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    if np.any(np.isnan(X_scaled)) or np.any(np.isnan(y_scaled)):
        print("Warning: Data contains NaN values after scaling, filling them")
        X_scaled = np.nan_to_num(X_scaled)
        y_scaled = np.nan_to_num(y_scaled)
    
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_scaled, y_scaled, indices, test_size=test_size, random_state=42
    )
    
    print(f"\nData preprocessing completed:")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of targets: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_indices, test_indices

# 3. 构建简化版的多输出神经网络模型
class SimpleMultiOutputRegressor(nn.Module):
    """Simplified multi-output neural network model"""
    def __init__(self, input_dim=6, output_dim=2):
        super(SimpleMultiOutputRegressor, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(16, 8)
        
        self.fc4 = nn.Linear(8, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
        
        return x

# 4. 训练多输出模型
def train_multi_output_model(model, X_train, y_train, X_val, y_val, epochs=4000, batch_size=8):
    """Train multi-output model"""
    model = model.to(device)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 200
    patience_counter = 0
    best_model_state = None
    
    print("Start training multi-output model...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_inputs = torch.from_numpy(X_val).to(device)
            val_targets = torch.from_numpy(y_val).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'Training completed, best validation loss: {best_val_loss:.6f}')
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

# 5. 评估多输出模型
def evaluate_multi_output_model(model, X_test, y_test, scaler_y):
    """Evaluate multi-output model performance"""
    model.eval()
    model = model.to(device)
    
    X_test_tensor = torch.from_numpy(X_test).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    mae_cis = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_trans = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mae_avg = (mae_cis + mae_trans) / 2
    
    rmse_cis = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_trans = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    rmse_avg = (rmse_cis + rmse_trans) / 2
    
    r2_cis = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_trans = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_avg = (r2_cis + r2_trans) / 2
    
    print("=" * 60)
    print("Multi-output model evaluation results:")
    print(f"Cis - R²: {r2_cis:.4f}, MAE: {mae_cis:.4f}, RMSE: {rmse_cis:.4f}")
    print(f"Trans - R²: {r2_trans:.4f}, MAE: {mae_trans:.4f}, RMSE: {rmse_trans:.4f}")
    print(f"Average - R²: {r2_avg:.4f}, MAE: {mae_avg:.4f}, RMSE: {rmse_avg:.4f}")
    print("=" * 60)
    
    return y_true, y_pred, {
        'MAE_Cis': mae_cis, 'MAE_Trans': mae_trans, 'MAE_avg': mae_avg,
        'RMSE_Cis': rmse_cis, 'RMSE_Trans': rmse_trans, 'RMSE_avg': rmse_avg,
        'R2_Cis': r2_cis, 'R2_Trans': r2_trans, 'R2_avg': r2_avg
    }

# 6. 计算Cis:Trans比值（改进版）
def calculate_ratio(cis, trans):
    """计算Cis:Trans比值，处理各种边缘情况"""
    ratio = np.zeros_like(cis)
    
    for i in range(len(cis)):
        # 处理无效值
        if np.isnan(cis[i]) or np.isnan(trans[i]) or np.isinf(cis[i]) or np.isinf(trans[i]):
            ratio[i] = 0.0
            continue
            
        # 确保值为非负
        cis_val = max(cis[i], 0)
        trans_val = max(trans[i], 0)
        
        # 处理分母接近0的情况
        denominator = cis_val + trans_val
        
        if denominator < 1e-8:  # 分母太小，无法计算有意义的比值
            ratio[i] = 0.0
        elif cis_val < 0.2 and trans_val < 0.2:  # 两个值都太小
            ratio[i] = 0.0
        else:
            # 计算比值并限制在[-1, 1]范围内
            ratio[i] = (cis_val - trans_val) / denominator
            # 确保比值在合理范围内
    
    return ratio

# 7. 可视化多输出结果并保存PDF - 优化用于论文发表
def visualize_multi_output_results(y_true, y_pred, history=None, save_plots=True, plot_name="prediction_results"):
    """Visualize multi-output prediction results and training process - optimized for publication"""
    fig = plt.figure(figsize=(18, 12))
    
    # 设置颜色方案 - 适合论文发表的颜色
    cis_color = '#1f77b4'  # 蓝色 - Cis
    trans_color = '#ff7f0e'  # 橙色 - Trans
    ratio_color = '#2ca02c'  # 绿色 - Ratio
    line_color = '#d62728'  # 红色 - 理想预测线
    grid_color = '#e0e0e0'
    
    # 计算Cis和Trans的R²分数
    r2_cis = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_trans = r2_score(y_true[:, 1], y_pred[:, 1])
    
    # 1. Cis和Trans：真实值 vs 预测值散点图（合并在一个图上）
    plt.subplot(2, 3, 1)
    
    # 绘制Cis的预测结果
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=cis_color, label=f'Cis (R²={r2_cis:.4f})', s=60, marker='o')
    
    # 绘制Trans的预测结果
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=trans_color, label=f'Trans (R²={r2_trans:.4f})', s=60, marker='^')
    
    # 绘制理想预测线
    all_true = np.concatenate([y_true[:, 0], y_true[:, 1]])
    all_pred = np.concatenate([y_pred[:, 0], y_pred[:, 1]])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color=line_color, lw=2, label='Ideal prediction')
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Cis and Trans: True vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, color=grid_color, alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 2. Cis和Trans残差图（合并在一个图上）
    plt.subplot(2, 3, 2)
    residuals_cis = y_true[:, 0] - y_pred[:, 0]
    residuals_trans = y_true[:, 1] - y_pred[:, 1]
    
    # 绘制Cis残差
    plt.scatter(y_pred[:, 0], residuals_cis, alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=cis_color, label='Cis residuals', s=60, marker='o')
    # 绘制Trans残差
    plt.scatter(y_pred[:, 1], residuals_trans, alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=trans_color, label='Trans residuals', s=60, marker='^')
    
    plt.axhline(y=0, color=line_color, linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Cis and Trans Residual Plot', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, color=grid_color, alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Cis和Trans误差分布直方图（合并在一个图上）
    plt.subplot(2, 3, 3)
    
    # 计算误差
    errors_cis = y_true[:, 0] - y_pred[:, 0]
    errors_trans = y_true[:, 1] - y_pred[:, 1]
    
    # 绘制Cis误差分布
    plt.hist(errors_cis, bins=15, edgecolor='black', alpha=0.7, color=cis_color, 
             label=f'Cis (MAE={np.mean(np.abs(errors_cis)):.3f})', density=True, hatch='//')
    # 绘制Trans误差分布
    plt.hist(errors_trans, bins=15, edgecolor='black', alpha=0.7, color=trans_color, 
             label=f'Trans (MAE={np.mean(np.abs(errors_trans)):.3f})', density=True, hatch='\\\\')
    
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Cis and Trans Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, color=grid_color, alpha=0.7, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 4. Cis和Trans预测误差箱线图
    plt.subplot(2, 3, 4)
    
    box_data = [errors_cis, errors_trans]
    box_labels = ['Cis', 'Trans']
    
    box = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # 设置箱线图颜色
    colors = [cis_color, trans_color]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置中位数线颜色
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    plt.ylabel('Prediction Error', fontsize=12)
    plt.title('Cis and Trans Error Distribution Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, color=grid_color, alpha=0.7, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 5. 训练损失历史
    if history is not None:
        plt.subplot(2, 3, 5)
        plt.plot(history['train_loss'], label='Training Loss', linewidth=2, color=cis_color)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color=trans_color)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Model Training History', fontsize=14, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, color=grid_color, alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 6. Cis:Trans比值对比图
    plt.subplot(2, 3, 6)
    
    # 计算比值（使用新规则：Cis和Trans同时小于0.1时比值为0）
    ratio_true = calculate_ratio(y_true[:, 0], y_true[:, 1])
    ratio_pred = calculate_ratio(y_pred[:, 0], y_pred[:, 1])
    
    # 排除无效值
    valid_indices = ~(np.isnan(ratio_true) | np.isnan(ratio_pred) | np.isinf(ratio_true) | np.isinf(ratio_pred) | (ratio_true == 0))
    if np.sum(valid_indices) > 0:
        ratio_true_valid = ratio_true[valid_indices]
        ratio_pred_valid = ratio_pred[valid_indices]
        r2_ratio = r2_score(ratio_true_valid, ratio_pred_valid)
        
        print(f"R² for Cis:Trans ratio: {r2_ratio:.4f}")
        print(ratio_true_valid)
        print(ratio_pred_valid)
        print(y_true[:, 0], y_true[:, 1])
        print(y_pred[:, 0], y_pred[:, 1])
        
        # 保存ratio数据到CSV文件
        ratio_df = pd.DataFrame({
            'ratio_true': ratio_true_valid,
            'ratio_pred': ratio_pred_valid
        })
        ratio_csv_path = os.path.join(output_dir, "ratio_comparison_data.csv")
        ratio_df.to_csv(ratio_csv_path, index=False)
        print(f"Ratio comparison data saved to: {ratio_csv_path}")

    else:
        r2_ratio = np.nan
    
    # 绘制比值对比散点图
    plt.scatter(ratio_true_valid, ratio_pred_valid, alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=ratio_color, label=f'Cis:Trans ratio (R²={r2_ratio:.4f})', s=60, marker='o')
    
    # 绘制理想预测线
    if len(ratio_true_valid) > 0:
        ratio_min = min(ratio_true_valid.min(), ratio_pred_valid.min())
        ratio_max = max(ratio_true_valid.max(), ratio_pred_valid.max())
        plt.plot([ratio_min, ratio_max], [ratio_min, ratio_max], '--', color=line_color, lw=2, label='Ideal prediction')
    
    plt.xlabel('True Cis:Trans Ratio', fontsize=12)
    plt.ylabel('Predicted Cis:Trans Ratio', fontsize=12)
    plt.title('Cis:Trans Ratio Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, color=grid_color, alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # 保存为PDF（高分辨率）
    if save_plots:
        pdf_path = os.path.join(output_dir, f"{plot_name}.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', 
                    pad_inches=0.1, transparent=False)
        print(f"Plot saved as PDF: {pdf_path}")
    
    plt.show()
    
    # 打印一些统计信息
    print("\nPrediction result statistics:")
    print(f"Cis - Mean Absolute Error: {np.mean(np.abs(residuals_cis)):.4f}")
    print(f"Cis - Max Positive Error: {np.max(residuals_cis):.4f}")
    print(f"Cis - Max Negative Error: {np.min(residuals_cis):.4f}")
    print(f"Cis - Error Standard Deviation: {np.std(residuals_cis):.4f}")
    print(f"Trans - Mean Absolute Error: {np.mean(np.abs(residuals_trans)):.4f}")
    print(f"Trans - Max Positive Error: {np.max(residuals_trans):.4f}")
    print(f"Trans - Max Negative Error: {np.min(residuals_trans):.4f}")
    print(f"Trans - Error Standard Deviation: {np.std(residuals_trans):.4f}")
    
    # 打印比值统计信息
    if np.sum(valid_indices) > 0:
        print(f"\nCis:Trans Ratio statistics:")
        print(f"True ratio range: [{np.nanmin(ratio_true):.4f}, {np.nanmax(ratio_true):.4f}]")
        print(f"Predicted ratio range: [{np.nanmin(ratio_pred):.4f}, {np.nanmax(ratio_pred):.4f}]")
        print(f"Ratio R²: {r2_ratio:.4f}")
        print(f"Ratio MAE: {np.mean(np.abs(ratio_true_valid - ratio_pred_valid)):.4f}")
        
        # 计算有多少样本的比值被设置为0
        true_zero_ratio = np.sum(ratio_true == 0)
        pred_zero_ratio = np.sum(ratio_pred == 0)
        print(f"Samples with ratio=0 (true): {true_zero_ratio}/{len(ratio_true)} ({true_zero_ratio/len(ratio_true)*100:.1f}%)")
        print(f"Samples with ratio=0 (pred): {pred_zero_ratio}/{len(ratio_pred)} ({pred_zero_ratio/len(ratio_pred)*100:.1f}%)")
    
    return ratio_true, ratio_pred, r2_ratio

# 8. 多输出预测器类
class MultiOutputPredictor:
    """Multi-output predictor class"""
    
    def __init__(self, model_path=None):
        """Initialize predictor"""
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, X, y, feature_names, test_size=0.2):
        """Train multi-output model"""
        self.feature_names = feature_names
        
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_indices, test_indices = preprocess_data(X, y, test_size)
        
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        self.model = SimpleMultiOutputRegressor(input_dim=X.shape[1], output_dim=y.shape[1])
        print(f"Using simplified multi-output model architecture")
        print(f"Input dimension: {X.shape[1]}, Output dimension: {y.shape[1]}")
        
        print("\nModel architecture:")
        print(self.model)
        
        print("\nStart training model...")
        self.model, history = train_multi_output_model(
            self.model, X_train, y_train, X_test, y_test, 
            epochs=4000, batch_size=8
        )
        
        y_true, y_pred, metrics = evaluate_multi_output_model(self.model, X_test, y_test, self.scaler_y)
        
        ratio_true, ratio_pred, r2_ratio = visualize_multi_output_results(y_true, y_pred, history, save_plots=True, plot_name="prediction_results")
        
        return history, metrics, test_indices, ratio_true, ratio_pred, r2_ratio
    
    def predict(self, features):
        """Predict Cis and Trans"""
        if self.model is None or self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Model not trained or loaded, please train or load model first")
        
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        X_scaled = self.scaler_X.transform(features)
        
        input_tensor = torch.from_numpy(X_scaled).to(device)
        
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).cpu().numpy()
        
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction[0]
    
    def predict_with_ratio(self, features):
        """Predict Cis, Trans and calculate ratio"""
        y_pred = self.predict(features)
        cis, trans = y_pred[0], y_pred[1]
        
        # 计算比值（Cis和Trans同时小于0.1时比值为0）
        if cis < 0.2 or trans < 0.2:
            ratio = 0
        elif trans != 0:
            ratio = (cis - trans) / (cis + trans)
        else:
            ratio = 0
        
        return cis, trans, ratio
    
    def batch_predict(self, features_list):
        """Batch prediction"""
        predictions = []
        for features in features_list:
            pred = self.predict(features)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save_model(self, path='multi_output_predictor_simple'):
        """Save model"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_names': self.feature_names,
                'input_dim': self.model.fc1.in_features,
                'output_dim': self.model.fc4.out_features
            }, f'{path}_model.pth')
            
            import joblib
            joblib.dump(self.scaler_X, f'{path}_scaler_X.pkl')
            joblib.dump(self.scaler_y, f'{path}_scaler_y.pkl')
            print(f"Model saved to {path}_model.pth")
    
    def load_model(self, path='multi_output_predictor_simple'):
        """Load model"""
        checkpoint = torch.load(f'{path}_model.pth', map_location=device)
        
        input_dim = checkpoint.get('input_dim', 6)
        output_dim = checkpoint.get('output_dim', 2)
        
        self.model = SimpleMultiOutputRegressor(input_dim=input_dim, output_dim=output_dim)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        
        self.feature_names = checkpoint.get('feature_names', None)
        
        import joblib
        try:
            self.scaler_X = joblib.load(f'{path}_scaler_X.pkl')
            self.scaler_y = joblib.load(f'{path}_scaler_y.pkl')
            print(f"Model loaded from {path}_model.pth")
        except:
            print(f"Warning: Cannot load scalers")
        
        if self.feature_names:
            print(f"Features: {self.feature_names}")

# 9. RGB通道可视化并保存PDF - 优化用于论文发表
def visualize_rgb_channels(X, feature_names, y, save_plots=True):
    """Visualize RGB channel data - optimized for publication"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 设置颜色方案
    cis_color = '#1f77b4'  # 蓝色 - Cis
    trans_color = '#ff7f0e'  # 橙色 - Trans
    
    for i, (ax, feature) in enumerate(zip(axes[:min(6, len(feature_names))], feature_names[:6])):
        ax.scatter(X[:, i], y[:, 0], alpha=0.6, label='Cis', color=cis_color, s=30, marker='o')
        ax.scatter(X[:, i], y[:, 1], alpha=0.6, label='Trans', color=trans_color, s=30, marker='^')
        
        ax.set_xlabel(f'{feature} value', fontsize=11)
        ax.set_ylabel('Concentration', fontsize=11)
        ax.set_title(f'{feature} vs Cis/Trans', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, color='#e0e0e0', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    if len(feature_names) >= 6:
        ax = axes[5]
        box_data = [X[:, i] for i in range(6)]
        box = ax.boxplot(box_data, labels=feature_names[:6], patch_artist=True)
        
        # 设置箱线图颜色
        for patch in box['boxes']:
            patch.set_facecolor('#aec7e8')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('RGB Channel', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('RGB Channel Value Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, color='#e0e0e0', alpha=0.7, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    
    # 保存为PDF
    if save_plots:
        pdf_path = os.path.join(output_dir, "rgb_channels_visualization.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    pad_inches=0.1, transparent=False)
        print(f"RGB visualization plot saved as PDF: {pdf_path}")
    
    plt.show()

# 10. 特征相关性分析并保存PDF - 优化用于论文发表
def analyze_feature_correlation(df, rgb_columns, save_plots=True):
    """Analyze correlation between RGB features and target variables - optimized for publication"""
    # 重命名列以便在图中显示
    df_renamed = df.copy()
    if 'Y1' in df_renamed.columns:
        df_renamed = df_renamed.rename(columns={'Y1': 'Cis'})
    if 'Y2' in df_renamed.columns:
        df_renamed = df_renamed.rename(columns={'Y2': 'Trans'})
    
    corr_matrix = df_renamed[rgb_columns + ['Cis', 'Trans']].corr()
    
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn的热图，但自定义样式
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 创建自定义的颜色映射
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    
    plt.title('Correlation Heatmap: RGB Features vs Target Variables', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 保存为PDF
    if save_plots:
        pdf_path = os.path.join(output_dir, "feature_correlation_heatmap.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    pad_inches=0.1, transparent=False)
        print(f"Correlation heatmap saved as PDF: {pdf_path}")
    
    plt.show()
    
    # 重命名以便打印
    if 'Y1' in df.columns:
        y1_corr = corr_matrix['Cis'].drop(['Cis', 'Trans']).abs().sort_values(ascending=False)
    else:
        y1_corr = corr_matrix['Y1'].drop(['Y1', 'Y2']).abs().sort_values(ascending=False)
        
    if 'Y2' in df.columns:
        y2_corr = corr_matrix['Trans'].drop(['Cis', 'Trans']).abs().sort_values(ascending=False)
    else:
        y2_corr = corr_matrix['Y2'].drop(['Y1', 'Y2']).abs().sort_values(ascending=False)
    
    print("\nRGB features with highest correlation to Cis:")
    for feature, corr in y1_corr.head(5).items():
        print(f"  {feature}: {corr:.4f}")
    
    print("\nRGB features with highest correlation to Trans:")
    for feature, corr in y2_corr.head(5).items():
        print(f"  {feature}: {corr:.4f}")

# 11. 创建训练历史单独图表并保存PDF - 优化用于论文发表
def create_training_history_plot(history, save_plots=True):
    """Create separate plot for training history - optimized for publication"""
    if history is not None and 'train_loss' in history and 'val_loss' in history:
        plt.figure(figsize=(10, 6))
        
        # 设置颜色
        train_color = '#1f77b4'  # 蓝色
        val_color = '#ff7f0e'  # 橙色
        
        plt.plot(history['train_loss'], label='Training Loss', linewidth=2.5, color=train_color)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2.5, color=val_color)
        
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Model Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, color='#e0e0e0', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # 设置y轴为科学计数法（如果需要）
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # 保存为PDF
        if save_plots:
            pdf_path = os.path.join(output_dir, "training_history.pdf")
            plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', 
                        pad_inches=0.1, transparent=False)
            print(f"Training history plot saved as PDF: {pdf_path}")
        
        plt.show()

# 12. 创建性能指标总结表并保存PDF - 优化用于论文发表
def create_performance_summary_table(metrics, r2_ratio=None, save_plots=True):
    """Create performance summary table as a figure - optimized for publication"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # 创建表格数据
    table_data = [
        ['Metric', 'Cis', 'Trans', 'Average'],
        ['R² Score', f"{metrics['R2_Cis']:.4f}", f"{metrics['R2_Trans']:.4f}", f"{metrics['R2_avg']:.4f}"],
        ['MAE', f"{metrics['MAE_Cis']:.4f}", f"{metrics['MAE_Trans']:.4f}", f"{metrics['MAE_avg']:.4f}"],
        ['RMSE', f"{metrics['RMSE_Cis']:.4f}", f"{metrics['RMSE_Trans']:.4f}", f"{metrics['RMSE_avg']:.4f}"]
    ]
    
    if r2_ratio is not None:
        table_data.append(['Cis:Trans Ratio R²', f"{r2_ratio:.4f}", "", ""])
    
    # 创建表格
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置表格样式 - 使用适合论文的配色
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # 标题行
                table[(i, j)].set_facecolor('#4F81BD')  # 深蓝色
                table[(i, j)].set_text_props(weight='bold', color='white', fontsize=12)
            elif i % 2 == 0:  # 偶数行
                table[(i, j)].set_facecolor('#D9E1F2')  # 浅蓝色
            else:  # 奇数行
                table[(i, j)].set_facecolor('#FFFFFF')  # 白色
            
            # 设置单元格边框
            table[(i, j)].set_edgecolor('#BFBFBF')
            table[(i, j)].set_linewidth(0.5)
    
    ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 保存为PDF
    if save_plots:
        pdf_path = os.path.join(output_dir, "performance_summary.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    pad_inches=0.2, transparent=False)
        print(f"Performance summary saved as PDF: {pdf_path}")
    
    plt.show()

# 13. 创建Cis:Trans比值详细分析图并保存PDF - 优化用于论文发表
def create_ratio_detailed_analysis(y_true, y_pred, save_plots=True):
    """Create detailed analysis of Cis:Trans ratio prediction - optimized for publication"""
    # 计算比值（使用新规则：Cis和Trans同时小于0.1时比值为0）
    ratio_true = calculate_ratio(y_true[:, 0], y_true[:, 1])
    ratio_pred = calculate_ratio(y_pred[:, 0], y_pred[:, 1])
    
    # 排除无效值
    valid_indices = ~(np.isnan(ratio_true) | np.isnan(ratio_pred) | np.isinf(ratio_true) | np.isinf(ratio_pred) | (ratio_true == 0))
    ratio_true_valid = ratio_true[valid_indices]
    ratio_pred_valid = ratio_pred[valid_indices]
    
    # 计算R²分数
    r2_ratio = r2_score(ratio_true_valid, ratio_pred_valid) if len(ratio_true_valid) > 0 else np.nan
    
    # 计算有多少样本的比值被设置为0
    true_zero_ratio = np.sum(ratio_true == 0)
    pred_zero_ratio = np.sum(ratio_pred == 0)
    
    # 设置颜色方案
    ratio_color = '#2ca02c'  # 绿色
    error_color = '#d62728'  # 红色
    true_color = '#1f77b4'  # 蓝色
    pred_color = '#ff7f0e'  # 橙色
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 比值散点图
    ax = axes[0, 0]
    scatter = ax.scatter(ratio_true_valid, ratio_pred_valid, alpha=0.7, edgecolors='k', 
                         linewidth=0.5, c=ratio_true_valid, cmap='viridis', s=80, marker='o')
    
    # 绘制理想预测线
    if len(ratio_true_valid) > 0:
        ratio_min = min(ratio_true_valid.min(), ratio_pred_valid.min())
        ratio_max = max(ratio_true_valid.max(), ratio_pred_valid.max())
        ax.plot([ratio_min, ratio_max], [ratio_min, ratio_max], '--', color=error_color, 
                lw=2.5, label='Ideal prediction')
    
    ax.set_xlabel('True Cis:Trans Ratio', fontsize=12)
    ax.set_ylabel('Predicted Cis:Trans Ratio', fontsize=12)
    ax.set_title(f'Cis:Trans Ratio Prediction (R²={r2_ratio:.4f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, color='#e0e0e0', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('True Ratio Value', fontsize=11)
    
    # 2. 比值误差分布
    ax = axes[0, 1]
    ratio_errors = ratio_pred_valid - ratio_true_valid
    
    ax.hist(ratio_errors, bins=20, edgecolor='black', alpha=0.7, color=ratio_color, hatch='//')
    ax.axvline(x=0, color=error_color, linestyle='--', linewidth=2.5)
    ax.set_xlabel('Ratio Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Cis:Trans Ratio Error Distribution\nMAE={np.mean(np.abs(ratio_errors)):.4f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, color='#e0e0e0', alpha=0.7, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 3. 比值误差 vs 真实比值
    ax = axes[1, 0]
    ax.scatter(ratio_true_valid, ratio_errors, alpha=0.7, edgecolors='k', linewidth=0.5, 
               color=error_color, s=60)
    ax.axhline(y=0, color=error_color, linestyle='--', linewidth=2.5)
    ax.set_xlabel('True Cis:Trans Ratio', fontsize=12)
    ax.set_ylabel('Ratio Prediction Error', fontsize=12)
    ax.set_title('Ratio Error vs True Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, color='#e0e0e0', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 4. 比值对比箱线图
    ax = axes[1, 1]
    box_data = [ratio_true_valid, ratio_pred_valid]
    box_labels = ['True Ratio', 'Predicted Ratio']
    
    box = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # 设置箱线图颜色
    colors = [true_color, pred_color]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置中位数线颜色
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Cis:Trans Ratio Value', fontsize=12)
    ax.set_title('True vs Predicted Ratio Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, color='#e0e0e0', alpha=0.7, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # 保存为PDF
    if save_plots:
        pdf_path = os.path.join(output_dir, "ratio_detailed_analysis.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', 
                    pad_inches=0.1, transparent=False)
        print(f"Ratio detailed analysis plot saved as PDF: {pdf_path}")
    
    plt.show()
    
    # 打印比值统计信息
    print(f"\nCis:Trans Ratio Detailed Statistics:")
    print(f"True ratio - Mean: {np.mean(ratio_true_valid):.4f}, Std: {np.std(ratio_true_valid):.4f}")
    print(f"Predicted ratio - Mean: {np.mean(ratio_pred_valid):.4f}, Std: {np.std(ratio_pred_valid):.4f}")
    print(f"Ratio R²: {r2_ratio:.4f}")
    print(f"Ratio MAE: {np.mean(np.abs(ratio_errors)):.4f}")
    print(f"Ratio RMSE: {np.sqrt(np.mean(ratio_errors**2)):.4f}")
    print(f"Samples with ratio=0 (true): {true_zero_ratio}/{len(ratio_true)} ({true_zero_ratio/len(ratio_true)*100:.1f}%)")
    print(f"Samples with ratio=0 (pred): {pred_zero_ratio}/{len(ratio_pred)} ({pred_zero_ratio/len(ratio_pred)*100:.1f}%)")
    
    return r2_ratio

# 14. 主函数
def main():
    """Main function"""
    print("=" * 60)
    print("Multi-output Neural Network Regression - Cis and Trans Prediction")
    print("RGB channels only")
    print(f"Device: {device}")
    print("=" * 60)
    
    print(f"\nOutput directory for plots: {output_dir}")
    
    X, y, feature_columns, df = create_dataset_from_csv()
    
    if device.type == 'cuda':
        print(f"\nGPU memory info:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print("\n" + "="*80)
    print("RGB Channel Visualization")
    print("="*80)
    visualize_rgb_channels(X, feature_columns, y, save_plots=True)
    
    print("\n" + "="*80)
    print("RGB Feature Correlation Analysis")
    print("="*80)
    analyze_feature_correlation(df, feature_columns, save_plots=True)
    
    print("\n" + "="*80)
    print("Training Multi-output Model")
    print("="*80)
    
    predictor = MultiOutputPredictor()
    history, metrics, test_indices, ratio_true, ratio_pred, r2_ratio = predictor.train(X, y, feature_columns)
    
    # 创建额外的单独图表
    create_training_history_plot(history, save_plots=True)
    create_performance_summary_table(metrics, r2_ratio, save_plots=True)
    
    # 创建Cis:Trans比值详细分析图
    print("\n" + "="*80)
    print("Cis:Trans Ratio Detailed Analysis")
    print("="*80)
    
    # 重新获取预测结果
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_indices, test_indices = preprocess_data(X, y)
    y_true, y_pred, _ = evaluate_multi_output_model(predictor.model, X_test, y_test, scaler_y)
    r2_ratio_detailed = create_ratio_detailed_analysis(y_true, y_pred, save_plots=True)
    
    # 保存模型
    predictor.save_model('multi_output_predictor_rgb_simple')
    
    print("\n" + "="*80)
    print("Test Prediction")
    print("="*80)
    
    if len(test_indices) > 0:
        test_idx = test_indices[0]
        sample_features = X[test_idx]
        actual_cis, actual_trans = y[test_idx]
        
        predicted = predictor.predict(sample_features)
        
        print(f"Test sample index: {test_idx}")
        print(f"RGB feature values: {sample_features}")
        print(f"Feature names: {feature_columns}")
        print(f"Actual values: Cis={actual_cis:.4f}, Trans={actual_trans:.4f}")
        print(f"Predicted values: Cis={predicted[0]:.4f}, Trans={predicted[1]:.4f}")
        print(f"Absolute errors: Cis={abs(actual_cis-predicted[0]):.4f}, Trans={abs(actual_trans-predicted[1]):.4f}")
        
        # 计算比值（使用新规则：Cis和Trans同时小于0.1时比值为0）
        if actual_cis < 0.1 and actual_trans < 0.1:
            actual_ratio = 0
        elif actual_trans != 0:
            actual_ratio = (actual_cis - actual_trans) / (actual_cis + actual_trans)
        else:
            actual_ratio = 0
        
        if predicted[0] < 0.1 and predicted[1] < 0.1:
            predicted_ratio = 0
        elif predicted[1] != 0:
            predicted_ratio = (predicted[0] - predicted[1]) / (predicted[0] + predicted[1])
        else:
            predicted_ratio = 0
        
        print(f"Actual Cis:Trans ratio: {actual_ratio:.4f}")
        print(f"Predicted Cis:Trans ratio: {predicted_ratio:.4f}")
        print(f"Ratio error: {abs(actual_ratio-predicted_ratio):.4f}")
    else:
        print("No test samples available")
    
    print("\n" + "="*80)
    print("Model Performance Summary")
    print("="*80)
    print(f"Final test performance:")
    print(f"  Cis R²: {metrics['R2_Cis']:.4f}, MAE: {metrics['MAE_Cis']:.4f}")
    print(f"  Trans R²: {metrics['R2_Trans']:.4f}, MAE: {metrics['MAE_Trans']:.4f}")
    print(f"  Average R²: {metrics['R2_avg']:.4f}, Average MAE: {metrics['MAE_avg']:.4f}")
    print(f"  Cis:Trans Ratio R²: {r2_ratio:.4f}")
    
    print(f"\nAll plots have been saved as PDF files in the '{output_dir}' directory.")
    print("These PDFs are optimized for editing in Adobe Illustrator:")
    print("- Text remains as editable text (not converted to curves)")
    print("- High resolution (300 DPI or 600 DPI)")
    print("- Arial font for maximum compatibility")
    print("- Color scheme suitable for publication")
    
    return predictor, X, y, metrics, r2_ratio

# 15. 运行主程序
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Multi-output Neural Network Regression - Cis and Trans Prediction")
        print("=" * 60)
        
        predictor, X, y, metrics, r2_ratio = main()
        
        print("\n" + "="*60)
        print("Program execution completed!")
        print(f"Final model average R² score: {metrics['R2_avg']:.4f}")
        print(f"Cis:Trans Ratio R²: {r2_ratio:.4f}")
        print(f"Model saved as 'multi_output_predictor_rgb_simple_model.pth'")
        print(f"\nPDF files saved in '{output_dir}' directory:")
        
        # 列出保存的PDF文件
        pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        
        print("\nUsage example:")
        print("""
        # Load multi-output model
        predictor = MultiOutputPredictor()
        predictor.load_model('multi_output_predictor_rgb_simple')
        
        # Prepare RGB feature data (6 values)
        rgb_features = [251, 225, 29, 250, 242, 86]  # Example data
        
        # Make prediction with ratio
        cis, trans, ratio = predictor.predict_with_ratio(rgb_features)
        print(f"Prediction results: Cis={cis:.4f}, Trans={trans:.4f}")
        print(f"Cis:Trans ratio: {ratio:.4f}")
        
        # Note: Ratio is 0 if both Cis and Trans are less than 0.1
        """)
        print("="*60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nSolutions:")
        print("1. If GPU memory is insufficient, code will automatically use CPU")
        print("2. Make sure you have correct PyTorch version: pip install torch torchvision")
        print("3. Make sure data file path is correct")
        print("4. If problem persists, try restarting Python environment completely")