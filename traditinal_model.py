import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置PDF输出为可编辑文本（不转曲）
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # 输出Type 42字体（可编辑）
matplotlib.rcParams['ps.fonttype'] = 42   # 输出Type 42字体（可编辑）
matplotlib.rcParams['font.family'] = 'Arial'  # 使用通用字体确保兼容性

# 设置随机种子以确保可重复性
np.random.seed(42)

print("Traditional Machine Learning for Cis and Trans Prediction Analysis")

# 1. 数据集创建函数 - 从CSV读取数据
def create_dataset_from_csv():
    """Create dataset from CSV file"""
    df = pd.read_csv('all_colors_data.csv')
    print(f"Successfully read data from CSV file, dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    
    # 检查列名，修正最后一列名称（可能存在重复）
    if df.columns[-1] == 'Y1' and 'Y1' in df.columns:
        # 如果最后一列也是Y1，重命名为Y2
        df = df.rename(columns={df.columns[-1]: 'Y2'})
        print("Renamed the last column to Y2")
    
    # 重命名目标变量为Cis和Trans
    df = df.rename(columns={'Y1': 'Cis', 'Y2': 'Trans'})
    
    # 只选择RGB通道：A_B, A_G, A_R, B_B, B_G, B_R
    rgb_columns = ['A_B', 'A_G', 'A_R', 'B_B', 'B_G', 'B_R']
    
    # 检查这些列是否存在
    missing_cols = [col for col in rgb_columns if col not in df.columns]
    if missing_cols:
        print(f"警告: 以下RGB列不存在: {missing_cols}")
        # 使用所有包含B,G,R的列
        rgb_columns = []
        for col in df.columns:
            if col.startswith('A_') and any(ch in col for ch in ['B', 'G', 'R']):
                rgb_columns.append(col)
            elif col.startswith('B_') and any(ch in col for ch in ['B', 'G', 'R']):
                rgb_columns.append(col)
    
    print(f"RGB特征列 ({len(rgb_columns)}个): {rgb_columns}")
    print(f"目标列: Cis, Trans")
    
    # 提取特征和目标
    X = df[rgb_columns].values.astype(np.float32)
    y = df[['Cis', 'Trans']].values.astype(np.float32)
    
    # 检查数据
    print(f"\nData statistics:")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Cis range: [{y[:, 0].min():.4f}, {y[:, 0].max():.4f}]")
    print(f"Trans range: [{y[:, 1].min():.4f}, {y[:, 1].max():.4f}]")
    
    return X, y, rgb_columns, df

# 2. 数据预处理
def preprocess_data(X, y, test_size=0.2):
    """Data preprocessing function"""
    # 标准化特征
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 标准化目标变量
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # 划分训练集和测试集
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_scaled, y_scaled, indices, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_indices, test_indices

# 计算Cis/Trans比值，处理特殊情况
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
        # elif cis_val < 0.2 and trans_val < 0.2:  # 两个值都太小
        #     ratio[i] = 0.0
        else:
            # 计算比值并限制在[-1, 1]范围内
            ratio[i] = (cis_val - trans_val) / denominator
            # 确保比值在合理范围内
    
    return ratio

# 3. 传统机器学习模型比较（多输出回归）
def train_traditional_ml_models(X_train, y_train, X_test, y_test, scaler_y):
    """Train multiple traditional ML models and compare"""
    
    # 定义要比较的模型（使用MultiOutputRegressor包装器处理多输出）
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale')),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    # 存储结果
    results = {}
    predictions = {}
    
    print("Training traditional ML models...")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_scaled = model.predict(X_test)
            
            # 反标准化
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_test)
            
            # 计算评估指标（分别计算Cis和Trans的指标）
            mae_cis = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
            mae_trans = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
            mae_avg = (mae_cis + mae_trans) / 2
            
            rmse_cis = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
            rmse_trans = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
            rmse_avg = (rmse_cis + rmse_trans) / 2
            
            r2_cis = r2_score(y_true[:, 0], y_pred[:, 0])
            r2_trans = r2_score(y_true[:, 1], y_pred[:, 1])
            r2_avg = (r2_cis + r2_trans) / 2
            
            # 计算比值指标
            cis_true_ratio = calculate_ratio(y_true[:, 0], y_true[:, 1])
            cis_pred_ratio = calculate_ratio(y_pred[:, 0], y_pred[:, 1])
            
            # 移除无穷大和NaN值
            valid_indices = ~(np.isnan(cis_true_ratio) | np.isnan(cis_pred_ratio) | np.isinf(cis_true_ratio) | np.isinf(cis_pred_ratio) | (cis_true_ratio == 0))
            if np.sum(valid_indices) > 0:
                r2_ratio = r2_score(cis_true_ratio[valid_indices], cis_pred_ratio[valid_indices])
                mae_ratio = mean_absolute_error(cis_true_ratio[valid_indices], cis_pred_ratio[valid_indices])
                rmse_ratio = np.sqrt(mean_squared_error(cis_true_ratio[valid_indices], cis_pred_ratio[valid_indices]))
                ratio_df = pd.DataFrame({
                    'ratio_true': cis_true_ratio[valid_indices],
                    'ratio_pred': cis_pred_ratio[valid_indices]
                })
                ratio_df.to_csv("trad_ratio_comparison_data.csv", index=False)
                print(f"Ratio comparison data saved to: trad_ratio_comparison_data.csv")
            else:
                r2_ratio = -1
                mae_ratio = -1
                rmse_ratio = -1
            
            # 存储结果
            results[name] = {
                'MAE_Cis': mae_cis, 'MAE_Trans': mae_trans, 'MAE_avg': mae_avg,
                'RMSE_Cis': rmse_cis, 'RMSE_Trans': rmse_trans, 'RMSE_avg': rmse_avg,
                'R2_Cis': r2_cis, 'R2_Trans': r2_trans, 'R2_avg': r2_avg,
                'R2_Ratio': r2_ratio, 'MAE_Ratio': mae_ratio, 'RMSE_Ratio': rmse_ratio
            }
            predictions[name] = (y_true, y_pred, cis_true_ratio, cis_pred_ratio)
            
            print(f"{name} - R²(Cis): {r2_cis:.4f}, R²(Trans): {r2_trans:.4f}, R²(avg): {r2_avg:.4f}")
            print(f"       R²(Ratio): {r2_ratio:.4f}, MAE(avg): {mae_avg:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    print("=" * 80)
    
    # 找到最佳模型（基于平均R²分数）
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['R2_avg'])
        best_result = results[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best average R² score: {best_result['R2_avg']:.4f}")
        print(f"Cis R²: {best_result['R2_Cis']:.4f}, Trans R²: {best_result['R2_Trans']:.4f}")
        print(f"Ratio R²: {best_result['R2_Ratio']:.4f}")
        print(f"Best average MAE: {best_result['MAE_avg']:.4f}")
        print(f"Best average RMSE: {best_result['RMSE_avg']:.4f}")
    else:
        best_model_name = None
        best_result = None
        print("No models successfully trained")
    
    return results, predictions, best_model_name

# 4. 交叉验证评估
def cross_validate_models(X, y):
    """Evaluate model performance using cross-validation"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    cv_results = {}
    
    print("\nStarting cross-validation evaluation...")
    print("=" * 60)
    
    for name, model in models.items():
        try:
            # 使用5折交叉验证，分别评估Cis和Trans
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            cis_scores = []
            trans_scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                cis_scores.append(r2_score(y_val[:, 0], y_pred[:, 0]))
                trans_scores.append(r2_score(y_val[:, 1], y_pred[:, 1]))
            
            cis_mean, cis_std = np.mean(cis_scores), np.std(cis_scores)
            trans_mean, trans_std = np.mean(trans_scores), np.std(trans_scores)
            avg_mean = (cis_mean + trans_mean) / 2
            
            cv_results[name] = {
                'Cis_R2_mean': cis_mean, 'Cis_R2_std': cis_std,
                'Trans_R2_mean': trans_mean, 'Trans_R2_std': trans_std,
                'Avg_R2_mean': avg_mean
            }
            
            print(f"{name}:")
            print(f"  Cis R² = {cis_mean:.4f} (±{cis_std:.4f})")
            print(f"  Trans R² = {trans_mean:.4f} (±{trans_std:.4f})")
            print(f"  Average R² = {avg_mean:.4f}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error in cross-validation for {name}: {e}")
            continue
    
    print("=" * 60)
    
    return cv_results

# 5. 可视化模型比较
def visualize_model_comparison(results, predictions, best_model_name, feature_names, save_pdf=False):
    """Visualize comparison of multiple models and save to PDF with editable text"""
    
    if not results:
        print("No results available for visualization")
        return
    
    # 设置图形大小和样式（适合论文发表）
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'font.family': 'Arial'  # 使用Arial字体确保兼容性
    })
    
    # 定义适合论文的颜色方案
    cis_color = '#2E86AB'  # 深蓝色
    trans_color = '#A23B72'  # 深紫色
    ratio_color = '#F18F01'  # 橙色
    ideal_line_color = '#2D3047'  # 深灰色
    scatter_alpha = 0.6
    
    fig = plt.figure(figsize=(20, 18))
    
    # 1. 模型性能比较柱状图（平均R²分数）
    plt.subplot(4, 4, 1)
    model_names = list(results.keys())
    r2_avg_scores = [results[name]['R2_avg'] for name in model_names]
    
    # 按R²分数排序
    sorted_indices = np.argsort(r2_avg_scores)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_r2_avg = [r2_avg_scores[i] for i in sorted_indices]
    
    bars = plt.bar(range(len(sorted_names)), sorted_r2_avg, color=cis_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Model')
    plt.ylabel('Average R² Score')
    plt.title('Model Performance Comparison (Average R² Score)', fontweight='bold')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.ylim([max(0, min(sorted_r2_avg) - 0.1), 1.0])
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, sorted_r2_avg):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Cis和Trans的R²分数比较
    plt.subplot(4, 4, 2)
    x_pos = np.arange(len(sorted_names))
    width = 0.35
    
    r2_cis_scores = [results[name]['R2_Cis'] for name in sorted_names]
    r2_trans_scores = [results[name]['R2_Trans'] for name in sorted_names]
    
    bars1 = plt.bar(x_pos - width/2, r2_cis_scores, width, label='Cis', color=cis_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x_pos + width/2, r2_trans_scores, width, label='Trans', color=trans_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison for Cis and Trans', fontweight='bold')
    plt.xticks(x_pos, sorted_names, rotation=45, ha='right')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.ylim([max(0, min(min(r2_cis_scores), min(r2_trans_scores)) - 0.1), 1.0])
    
    # 3. 最佳模型的预测结果（Cis和Trans在同一图上）
    if best_model_name:
        y_true, y_pred, cis_true_ratio, cis_pred_ratio = predictions[best_model_name]
        
        plt.subplot(4, 4, 3)
        # 绘制Cis的预测结果
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=cis_color, label='Cis', s=40)
        # 绘制Trans的预测结果
        plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=trans_color, label='Trans', s=40, marker='^')
        
        # 绘制理想预测线
        all_true = np.concatenate([y_true[:, 0], y_true[:, 1]])
        all_pred = np.concatenate([y_pred[:, 0], y_pred[:, 1]])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color=ideal_line_color, linestyle='--', lw=2, label='Ideal Prediction Line')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{best_model_name} - Cis and Trans: True vs Predicted', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # 添加文本标注R²分数
        cis_r2 = results[best_model_name]['R2_Cis']
        trans_r2 = results[best_model_name]['R2_Trans']
        plt.text(0.05, 0.95, f'Cis R² = {cis_r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(0.05, 0.88, f'Trans R² = {trans_r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. 残差图（Cis和Trans在同一图上）
    if best_model_name:
        plt.subplot(4, 4, 4)
        residuals_cis = y_true[:, 0] - y_pred[:, 0]
        residuals_trans = y_true[:, 1] - y_pred[:, 1]
        
        # 绘制Cis残差
        plt.scatter(y_pred[:, 0], residuals_cis, alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=cis_color, label='Cis', s=40)
        # 绘制Trans残差
        plt.scatter(y_pred[:, 1], residuals_trans, alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=trans_color, label='Trans', s=40, marker='^')
        
        plt.axhline(y=0, color=ideal_line_color, linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{best_model_name} - Residual Plot for Cis and Trans', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.2, linestyle='--')
    
    # 5. 误差分布直方图（Cis和Trans在同一图上）
    if best_model_name:
        plt.subplot(4, 4, 5)
        
        # 计算误差
        errors_cis = y_true[:, 0] - y_pred[:, 0]
        errors_trans = y_true[:, 1] - y_pred[:, 1]
        
        # 绘制Cis误差分布
        plt.hist(errors_cis, bins=15, edgecolor='black', alpha=0.6, color=cis_color, label='Cis', density=True, linewidth=0.5)
        # 绘制Trans误差分布
        plt.hist(errors_trans, bins=15, edgecolor='black', alpha=0.6, color=trans_color, label='Trans', density=True, linewidth=0.5)
        
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title(f'{best_model_name} - Error Distribution for Cis and Trans', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 6. 误差箱线图（Cis和Trans比较）
    if best_model_name:
        plt.subplot(4, 4, 6)
        errors_cis = y_true[:, 0] - y_pred[:, 0]
        errors_trans = y_true[:, 1] - y_pred[:, 1]
        
        box_data = [errors_cis, errors_trans]
        box_labels = ['Cis', 'Trans']
        
        box = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # 设置箱线图颜色
        colors = [cis_color, trans_color]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 设置异常点样式
        for flier in box['fliers']:
            flier.set(marker='o', color='gray', alpha=0.5)
        
        plt.ylabel('Prediction Error')
        plt.title(f'{best_model_name} - Error Distribution Comparison', fontweight='bold')
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 7. Cis/Trans比值预测图（新增）
    if best_model_name:
        plt.subplot(4, 4, 7)
        
        # 计算比值
        cis_true_ratio = calculate_ratio(y_true[:, 0], y_true[:, 1])
        cis_pred_ratio = calculate_ratio(y_pred[:, 0], y_pred[:, 1])
        
        # 移除无穷大和NaN值
        valid_indices = np.isfinite(cis_true_ratio) & np.isfinite(cis_pred_ratio)
        cis_true_ratio_valid = cis_true_ratio[valid_indices]
        cis_pred_ratio_valid = cis_pred_ratio[valid_indices]
        
        # 绘制比值预测图
        plt.scatter(cis_true_ratio_valid, cis_pred_ratio_valid, alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=ratio_color, s=50)
        
        # 绘制理想预测线
        min_val = min(cis_true_ratio_valid.min(), cis_pred_ratio_valid.min())
        max_val = max(cis_true_ratio_valid.max(), cis_pred_ratio_valid.max())
        plt.plot([min_val, max_val], [min_val, max_val], color=ideal_line_color, linestyle='--', lw=2, label='Ideal Prediction Line')
        
        # 计算并显示R²分数
        if len(cis_true_ratio_valid) > 0:
            r2_ratio = r2_score(cis_true_ratio_valid, cis_pred_ratio_valid)
            
            # 添加拟合线
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(cis_true_ratio_valid, cis_pred_ratio_valid)
            line_x = np.linspace(min_val, max_val, 100)
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, color=ratio_color, linestyle='-', lw=1.5, alpha=0.7, label=f'Fit Line (R²={r2_ratio:.3f})')
            
            plt.text(0.05, 0.95, f'R² = {r2_ratio:.3f}', transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('True Cis/Trans Ratio')
        plt.ylabel('Predicted Cis/Trans Ratio')
        plt.title(f'{best_model_name} - Cis/Trans Ratio Prediction', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.2, linestyle='--')
    
    # 8. 比值残差图
    if best_model_name:
        plt.subplot(4, 4, 8)
        
        # 计算比值残差
        ratio_residuals = cis_true_ratio_valid - cis_pred_ratio_valid
        
        plt.scatter(cis_pred_ratio_valid, ratio_residuals, alpha=scatter_alpha, edgecolors='k', linewidth=0.3, 
                   color=ratio_color, s=50)
        
        plt.axhline(y=0, color=ideal_line_color, linestyle='--', linewidth=2)
        plt.xlabel('Predicted Cis/Trans Ratio')
        plt.ylabel('Residuals')
        plt.title(f'{best_model_name} - Ratio Residual Plot', fontweight='bold')
        plt.grid(True, alpha=0.2, linestyle='--')
    
    # 9. 模型性能热图
    plt.subplot(4, 4, 9)
    
    # 准备热图数据
    metrics = ['R2_avg', 'MAE_avg', 'RMSE_avg']
    metric_labels = ['Avg R²', 'Avg MAE', 'Avg RMSE']
    
    heatmap_data = np.zeros((len(sorted_names), len(metrics)))
    
    for i, name in enumerate(sorted_names):
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = results[name][metric]
    
    # 归一化R²分数（越高越好）和误差（越低越好）
    normalized_data = np.zeros_like(heatmap_data)
    for j, metric in enumerate(metrics):
        if 'R2' in metric:
            # R²分数：归一化到0-1，越高越好
            normalized_data[:, j] = (heatmap_data[:, j] - heatmap_data[:, j].min()) / (heatmap_data[:, j].max() - heatmap_data[:, j].min())
        else:
            # MAE/RMSE：归一化到0-1，但需要反转（越低越好）
            normalized_data[:, j] = 1 - (heatmap_data[:, j] - heatmap_data[:, j].min()) / (heatmap_data[:, j].max() - heatmap_data[:, j].min())
    
    # 创建热图
    im = plt.imshow(normalized_data, cmap='RdYlGn', aspect='auto')
    
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    plt.title('Model Performance Heatmap', fontweight='bold')
    plt.xticks(range(len(metrics)), metric_labels)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.colorbar(im, label='Normalized Performance (1=best, 0=worst)')
    
    # 添加数值标签
    for i in range(len(sorted_names)):
        for j in range(len(metrics)):
            if 'R2' in metrics[j]:
                text = f'{heatmap_data[i, j]:.3f}'
            else:
                text = f'{heatmap_data[i, j]:.3f}'
            plt.text(j, i, text, ha='center', va='center', color='black' if normalized_data[i, j] > 0.5 else 'white', fontsize=8)
    
    # 10. 特征重要性分析（如果可用）
    plt.subplot(4, 4, 10)
    if best_model_name and ('Forest' in best_model_name or 'Boosting' in best_model_name):
        # 如果是树模型，可以获取特征重要性
        try:
            if best_model_name == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif best_model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # 简化：使用所有数据训练模型获取特征重要性
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)  # X需要在外部定义
            
            model.fit(X_scaled, y)  # y需要在外部定义
            importances = model.feature_importances_
            
            # 平均特征重要性（对多输出取平均）
            if len(importances.shape) > 1:
                importances = importances.mean(axis=0)
            
            indices = np.argsort(importances)[::-1]
            
            # 只显示前10个最重要的特征
            top_n = min(10, len(indices))
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            plt.bar(range(top_n), top_importances, color=cis_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title(f'{best_model_name} - Top {top_n} Feature Importance', fontweight='bold')
            plt.xticks(range(top_n), top_features, rotation=45, ha='right')
            plt.grid(True, alpha=0.2, linestyle='--', axis='y')
        except Exception as e:
            plt.text(0.5, 0.5, f'Cannot compute feature importance\n{str(e)[:30]}...', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance Analysis', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Feature importance available only for tree-based models\n(e.g., Random Forest, Gradient Boosting)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance Analysis', fontweight='bold')
    
    # 11. Cis预测值与真实值对比（时间序列风格）
    if best_model_name:
        plt.subplot(4, 4, 11)
        
        # 排序样本以便更好地可视化
        sorted_idx = np.argsort(y_true[:, 0])
        sorted_true_cis = y_true[sorted_idx, 0]
        sorted_pred_cis = y_pred[sorted_idx, 0]
        
        plt.plot(range(len(sorted_true_cis)), sorted_true_cis, color=cis_color, alpha=0.8, label='True Cis', linewidth=2)
        plt.plot(range(len(sorted_pred_cis)), sorted_pred_cis, color=cis_color, alpha=0.6, label='Predicted Cis', linewidth=1.5, linestyle='--')
        
        plt.xlabel('Sample Index (sorted by Cis)')
        plt.ylabel('Cis Value')
        plt.title(f'{best_model_name} - Cis Prediction Trend', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.2, linestyle='--')
    
    # 12. Trans预测值与真实值对比（时间序列风格）
    if best_model_name:
        plt.subplot(4, 4, 12)
        
        # 排序样本以便更好地可视化
        sorted_idx = np.argsort(y_true[:, 1])
        sorted_true_trans = y_true[sorted_idx, 1]
        sorted_pred_trans = y_pred[sorted_idx, 1]
        
        plt.plot(range(len(sorted_true_trans)), sorted_true_trans, color=trans_color, alpha=0.8, label='True Trans', linewidth=2)
        plt.plot(range(len(sorted_pred_trans)), sorted_pred_trans, color=trans_color, alpha=0.6, label='Predicted Trans', linewidth=1.5, linestyle='--')
        
        plt.xlabel('Sample Index (sorted by Trans)')
        plt.ylabel('Trans Value')
        plt.title(f'{best_model_name} - Trans Prediction Trend', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.2, linestyle='--')
    
    # 13. Cis和Trans比值分布图
    if best_model_name:
        plt.subplot(4, 4, 13)
        
        # 计算比值
        cis_true_ratio = calculate_ratio(y_true[:, 0], y_true[:, 1])
        cis_pred_ratio = calculate_ratio(y_pred[:, 0], y_pred[:, 1])
        
        # 移除无穷大和NaN值
        valid_indices = np.isfinite(cis_true_ratio) & np.isfinite(cis_pred_ratio)
        cis_true_ratio_valid = cis_true_ratio[valid_indices]
        cis_pred_ratio_valid = cis_pred_ratio[valid_indices]
        
        # 绘制比值分布
        plt.hist(cis_true_ratio_valid, bins=20, edgecolor='black', alpha=0.6, color=cis_color, label='True Ratio', density=True, linewidth=0.5)
        plt.hist(cis_pred_ratio_valid, bins=20, edgecolor='black', alpha=0.6, color=trans_color, label='Predicted Ratio', density=True, linewidth=0.5)
        
        plt.xlabel('Cis/Trans Ratio')
        plt.ylabel('Density')
        plt.title(f'{best_model_name} - Ratio Distribution Comparison', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 14. 比值误差箱线图
    if best_model_name:
        plt.subplot(4, 4, 14)
        
        # 计算比值误差
        ratio_errors = cis_true_ratio_valid - cis_pred_ratio_valid
        
        box = plt.boxplot([ratio_errors], labels=['Cis/Trans Ratio'], patch_artist=True)
        
        # 设置箱线图颜色
        box['boxes'][0].set_facecolor(ratio_color)
        box['boxes'][0].set_alpha(0.7)
        
        # 设置异常点样式
        for flier in box['fliers']:
            flier.set(marker='o', color='gray', alpha=0.5)
        
        plt.ylabel('Prediction Error')
        plt.title(f'{best_model_name} - Ratio Error Distribution', fontweight='bold')
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 15. 模型R²分数对比（Cis, Trans, Ratio）
    plt.subplot(4, 4, 15)
    
    x_pos = np.arange(len(sorted_names))
    width = 0.25
    
    r2_cis_scores = [results[name]['R2_Cis'] for name in sorted_names]
    r2_trans_scores = [results[name]['R2_Trans'] for name in sorted_names]
    r2_ratio_scores = [max(0, results[name]['R2_Ratio']) for name in sorted_names]  # 避免负值
    
    bars1 = plt.bar(x_pos - width, r2_cis_scores, width, label='Cis', color=cis_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x_pos, r2_trans_scores, width, label='Trans', color=trans_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = plt.bar(x_pos + width, r2_ratio_scores, width, label='Ratio', color=ratio_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison (Cis, Trans, Ratio)', fontweight='bold')
    plt.xticks(x_pos, sorted_names, rotation=45, ha='right')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.ylim([max(0, min(min(r2_cis_scores), min(r2_trans_scores), min(r2_ratio_scores)) - 0.1), 1.0])
    
    # 16. 最终模型选择建议
    plt.subplot(4, 4, 16)
    
    if best_model_name:
        # 显示最佳模型信息
        best_result = results[best_model_name]
        
        info_text = f"Best Model: {best_model_name}\n\n"
        info_text += f"Cis R²: {best_result['R2_Cis']:.4f}\n"
        info_text += f"Trans R²: {best_result['R2_Trans']:.4f}\n"
        info_text += f"Ratio R²: {best_result['R2_Ratio']:.4f}\n"
        info_text += f"Avg R²: {best_result['R2_avg']:.4f}\n"
        info_text += f"Avg MAE: {best_result['MAE_avg']:.4f}\n"
        info_text += f"Avg RMSE: {best_result['RMSE_avg']:.4f}"
        
        plt.text(0.5, 0.5, info_text, ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.title('Best Model Performance Summary', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No best model available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Model Selection Summary', fontweight='bold')
    
    plt.axis('off')
    
    plt.suptitle('Traditional Machine Learning Models for Cis and Trans Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存为PDF（确保文字可编辑）
    if save_pdf:
        pdf_filename = 'model_comparison_results_editable.pdf'
        # 使用bbox_inches='tight'但确保文字不转曲
        plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nSaved model comparison plot to {pdf_filename}")
        print("Note: Text in PDF is editable (not converted to outlines)")
    else:
        plt.show()
    
    # 打印详细统计信息
    print("\n" + "="*80)
    print("Detailed Model Performance Comparison:")
    print("="*80)
    
    for name in sorted_names:
        result = results[name]
        print(f"\n{name}:")
        print(f"  Cis - R²: {result['R2_Cis']:.4f}, MAE: {result['MAE_Cis']:.4f}, RMSE: {result['RMSE_Cis']:.4f}")
        print(f"  Trans - R²: {result['R2_Trans']:.4f}, MAE: {result['MAE_Trans']:.4f}, RMSE: {result['RMSE_Trans']:.4f}")
        print(f"  Ratio - R²: {result['R2_Ratio']:.4f}, MAE: {result['MAE_Ratio']:.4f}, RMSE: {result['RMSE_Ratio']:.4f}")
        print(f"  Average - R²: {result['R2_avg']:.4f}, MAE: {result['MAE_avg']:.4f}, RMSE: {result['RMSE_avg']:.4f}")
    
    print("\n" + "="*80)
    if best_model_name:
        print(f"Best model: {best_model_name}")
        print(f"Best average R² score: {results[best_model_name]['R2_avg']:.4f}")
        print(f"Best average MAE: {results[best_model_name]['MAE_avg']:.4f}")
        print(f"Best ratio R² score: {results[best_model_name]['R2_Ratio']:.4f}")
    print("="*80)
    
    # 关闭图形以释放内存
    plt.close(fig)

# 6. 特征相关性分析
def analyze_feature_correlation(df, feature_columns, save_pdf=False):
    """Analyze correlation between features and target variables with editable text"""
    # 计算相关性矩阵
    corr_matrix = df[feature_columns + ['Cis', 'Trans']].corr()
    
    # 绘制相关性热图
    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Features and Target Variables (Cis and Trans)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存为PDF（确保文字可编辑）
    if save_pdf:
        pdf_filename = 'feature_correlation_heatmap_editable.pdf'
        plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nSaved correlation heatmap to {pdf_filename}")
        print("Note: Text in PDF is editable (not converted to outlines)")
    else:
        plt.show()
    
    # 打印与Cis和Trans相关性最高的特征
    cis_corr = corr_matrix['Cis'].drop(['Cis', 'Trans']).abs().sort_values(ascending=False)
    trans_corr = corr_matrix['Trans'].drop(['Cis', 'Trans']).abs().sort_values(ascending=False)
    
    print("\nTop features most correlated with Cis:")
    for feature, corr in cis_corr.head(5).items():
        print(f"  {feature}: {corr:.4f}")
    
    print("\nTop features most correlated with Trans:")
    for feature, corr in trans_corr.head(5).items():
        print(f"  {feature}: {corr:.4f}")
    
    # 关闭图形以释放内存
    plt.close(fig)

# 7. 专门绘制Cis/Trans比值图的函数
def plot_cis_trans_ratio(results, predictions, best_model_name, save_pdf=False):
    """Create a dedicated plot for Cis/Trans ratio analysis"""
    
    if not best_model_name:
        print("No best model available for ratio plotting")
        return
    
    y_true, y_pred, cis_true_ratio, cis_pred_ratio = predictions[best_model_name]
    
    # 设置图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 定义颜色
    cis_color = '#2E86AB'  # 深蓝色
    trans_color = '#A23B72'  # 深紫色
    ratio_color = '#F18F01'  # 橙色
    ideal_line_color = '#2D3047'  # 深灰色
    
    # 1. Cis/Trans比值预测图（主图）
    ax1 = axes[0, 0]
    
    # 移除无穷大和NaN值
    valid_indices = np.isfinite(cis_true_ratio) & np.isfinite(cis_pred_ratio)
    cis_true_ratio_valid = cis_true_ratio[valid_indices]
    cis_pred_ratio_valid = cis_pred_ratio[valid_indices]
    
    # 绘制散点图
    ax1.scatter(cis_true_ratio_valid, cis_pred_ratio_valid, alpha=0.7, edgecolors='k', linewidth=0.3, 
               color=ratio_color, s=60)
    
    # 绘制理想预测线
    min_val = min(cis_true_ratio_valid.min(), cis_pred_ratio_valid.min())
    max_val = max(cis_true_ratio_valid.max(), cis_pred_ratio_valid.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color=ideal_line_color, linestyle='--', 
             lw=2.5, label='Ideal Prediction Line')
    
    # 计算并添加拟合线
    from scipy import stats
    if len(cis_true_ratio_valid) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(cis_true_ratio_valid, cis_pred_ratio_valid)
        r2_ratio = r_value**2
        line_x = np.linspace(min_val, max_val, 100)
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, color=ratio_color, linestyle='-', lw=2, alpha=0.8, 
                label=f'Linear Fit (R²={r2_ratio:.3f})')
        
        # 添加R²文本
        ax1.text(0.05, 0.95, f'R² = {r2_ratio:.3f}', transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel('True Cis/Trans Ratio', fontweight='bold')
    ax1.set_ylabel('Predicted Cis/Trans Ratio', fontweight='bold')
    ax1.set_title(f'{best_model_name} - Cis/Trans Ratio Prediction', fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # 2. 比值残差图
    ax2 = axes[0, 1]
    
    # 计算比值残差
    ratio_residuals = cis_true_ratio_valid - cis_pred_ratio_valid
    
    ax2.scatter(cis_pred_ratio_valid, ratio_residuals, alpha=0.7, edgecolors='k', linewidth=0.3, 
               color=ratio_color, s=60)
    
    ax2.axhline(y=0, color=ideal_line_color, linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Cis/Trans Ratio', fontweight='bold')
    ax2.set_ylabel('Residuals', fontweight='bold')
    ax2.set_title(f'{best_model_name} - Ratio Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # 3. 比值分布图
    ax3 = axes[1, 0]
    
    # 绘制比值分布
    ax3.hist(cis_true_ratio_valid, bins=20, edgecolor='black', alpha=0.6, color=cis_color, 
             label='True Ratio', density=True, linewidth=0.5)
    ax3.hist(cis_pred_ratio_valid, bins=20, edgecolor='black', alpha=0.6, color=trans_color, 
             label='Predicted Ratio', density=True, linewidth=0.5)
    
    ax3.set_xlabel('Cis/Trans Ratio', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title(f'{best_model_name} - Ratio Distribution', fontsize=13, fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 4. 比值误差箱线图
    ax4 = axes[1, 1]
    
    box = ax4.boxplot([ratio_residuals], labels=['Cis/Trans Ratio'], patch_artist=True)
    
    # 设置箱线图颜色
    box['boxes'][0].set_facecolor(ratio_color)
    box['boxes'][0].set_alpha(0.7)
    
    # 设置异常点样式
    for flier in box['fliers']:
        flier.set(marker='o', color='gray', alpha=0.5)
    
    ax4.set_ylabel('Prediction Error', fontweight='bold')
    ax4.set_title(f'{best_model_name} - Ratio Error Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    plt.suptitle('Cis/Trans Ratio Analysis for Best Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存为PDF（确保文字可编辑）
    if save_pdf:
        pdf_filename = 'cis_trans_ratio_analysis_editable.pdf'
        plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nSaved Cis/Trans ratio analysis plot to {pdf_filename}")
        print("Note: Text in PDF is editable (not converted to outlines)")
    else:
        plt.show()
    
    plt.close(fig)

# 主函数
if __name__ == "__main__":
    # 设置是否保存PDF
    save_pdf = True  # 设置为True以保存PDF，False以显示图形
    
    # 1. 从CSV创建数据集
    X, y, feature_columns, df = create_dataset_from_csv()
    
    # 2. 特征相关性分析
    print("\n" + "="*80)
    print("Feature Correlation Analysis")
    print("="*80)
    analyze_feature_correlation(df, feature_columns, save_pdf=save_pdf)
    
    # 3. 预处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_indices, test_indices = preprocess_data(X, y)
    
    print("\n" + "="*80)
    print("Traditional ML Model Performance Comparison")
    print("="*80)
    
    # 4. 训练并比较多种传统机器学习模型
    results, predictions, best_model_name = train_traditional_ml_models(
        X_train, y_train, X_test, y_test, scaler_y
    )
    
    # 5. 可视化模型比较结果
    visualize_model_comparison(results, predictions, best_model_name, feature_columns, save_pdf=save_pdf)
    
    # 6. 交叉验证评估
    cv_results = cross_validate_models(X, y)
    
    # 7. 专门绘制Cis/Trans比值图
    print("\n" + "="*80)
    print("Cis/Trans Ratio Analysis")
    print("="*80)
    plot_cis_trans_ratio(results, predictions, best_model_name, save_pdf=save_pdf)
    
    # 8. 数据统计信息
    print("\n" + "="*80)
    print("Data Statistics")
    print("="*80)
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Target variables: Cis, Trans (multi-output)")
    
    # 9. 模型选择建议
    print("\n" + "="*80)
    print("Model Selection Recommendations")
    print("="*80)
    
    if results:
        # 按平均R²分数排序
        sorted_models = sorted(results.items(), key=lambda x: x[1]['R2_avg'], reverse=True)
        
        print("Based on average R² scores and cross-validation results, recommend using the following models:")
        for i, (name, metrics) in enumerate(sorted_models[:3]):
            print(f"{i+1}. {name}:")
            print(f"   Average R² = {metrics['R2_avg']:.4f}, Average MAE = {metrics['MAE_avg']:.4f}")
            print(f"   Cis R² = {metrics['R2_Cis']:.4f}, Trans R² = {metrics['R2_Trans']:.4f}")
            print(f"   Ratio R² = {metrics['R2_Ratio']:.4f}")
    else:
        print("No model results available")
    
    print("\nFor multi-output regression problems, Random Forest or Gradient Boosting are recommended.")
    print("For high-dimensional feature spaces, regularized regression models (e.g., Ridge) can be considered.")