import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
h5_path = '/home/luhc/epigenetic/epigenetic/filtered_LGG_train_test_data.h5'

try:
    X_train_filtered = pd.read_hdf(h5_path, key='X_train_filtered')
    X_test_filtered = pd.read_hdf(h5_path, key='X_test_filtered')
    Y_train = pd.read_hdf(h5_path, key='Y_train')
    Y_test = pd.read_hdf(h5_path, key='Y_test')
except Exception as e:
    print(f"错误：加载数据失败。请确认路径和 key 是否正确。错误信息：{e}")
    exit()

imputer = SimpleImputer(strategy='mean')

# 2. 在训练集上 FIT (学习) 均值
# 这一步计算出 X_train_filtered 中每一列的平均值
imputer.fit(X_train_filtered) 

# 3. 应用 TRANSFORM (转换) 到训练集和测试集
# 使用学习到的训练集均值来填充 NaN

X_train_imputed = imputer.transform(X_train_filtered)
X_test_imputed = imputer.transform(X_test_filtered) 
X_train_filtered = pd.DataFrame(
    X_train_imputed, 
    index=X_train_filtered.index, 
    columns=X_train_filtered.columns
)
X_test_filtered = pd.DataFrame(
    X_test_imputed, 
    index=X_test_filtered.index, 
    columns=X_test_filtered.columns
)   
    
    
n_components=100
pca=PCA(n_components=n_components,random_state=42)
pca.fit(X_train_filtered)
X_train_pca=pca.transform(X_train_filtered)
X_test_pca=pca.transform(X_test_filtered)
pca_cols = [f'PC{i+1}' for i in range(n_components)]

X_train_pca = pd.DataFrame(
    X_train_pca, 
    index=X_train_filtered.index, 
    columns=pca_cols
)

X_test_pca = pd.DataFrame(
    X_test_pca, 
    index=X_test_filtered.index, 
    columns=pca_cols
)
print("\n--- PCA 降维完成 ---")
print(f"训练集特征形状 (降维后): {X_train_pca.shape}")
print(f"测试集特征形状 (降维后): {X_test_pca.shape}")
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"\n前 {n_components} 个主成分总共解释了 {explained_variance*100:.2f}% 的方差。")

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='.')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance (Cumulative)')
plt.grid(True)
plt.savefig('pca_variance.pdf', format='pdf', bbox_inches='tight')

# --- 关键修复步骤：统一取 Y 的第一列（生存时间） ---
Y_train_time = Y_train.iloc[:, 0]
Y_test_time = Y_test.iloc[:, 0]

# 4. 随机森林回归训练
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_pca, Y_train_time)

# 5. 在测试集上预测
Y_pred = rf_model.predict(X_test_pca)

# 6. 计算评估指标 (使用对齐后的 Y_test_time)
mse = mean_squared_error(Y_test_time, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_time, Y_pred)
r2 = r2_score(Y_test_time, Y_pred)

print("\n--- 模型评估结果 ---")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"R² 分数 (解释性评分): {r2:.4f}")

# --- 7. 可视化（加入生存状态颜色） ---
plt.figure(figsize=(8, 8))

# 提取生存状态列用于着色
# 假设 Y_test 的第二列是状态（0为存活/截断，1为死亡）
Y_test_status = Y_test.iloc[:, 1]

# 使用 c 参数指定颜色来源，cmap 指定色板（如 'RdYlGn_r'：红色代表死亡，绿色代表存活）
scatter = plt.scatter(Y_test_time, Y_pred, alpha=0.6, 
                      c=Y_test_status, cmap='RdYlGn_r', 
                      edgecolors='w', linewidth=0.5)

# 画出 y=x 的对角参考线
all_vals = np.concatenate([Y_test_time.values, Y_pred])
max_val = all_vals.max()
min_val = all_vals.min()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

# 添加图例说明颜色意义
handles, labels = scatter.legend_elements()
legend1 = plt.legend(handles, ['Alive/Censored (0)', 'Deceased (1)'], 
                    loc="upper left", title="Status")
plt.gca().add_artist(legend1)
plt.legend([plt.Line2D([0], [0], color='r', linestyle='--')], ['Perfect Prediction'], loc="lower right")

plt.xlabel('Actual Survival Time')
plt.ylabel('Predicted Survival Time')
plt.title('Actual vs Predicted Survival Time (Test Set)\nColored by Survival Status')
plt.grid(True, linestyle=':', alpha=0.6)

# 保存图片
plt.savefig('prediction_scatter_status.pdf', format='pdf', bbox_inches='tight')
print("\n预测结果图（已按状态着色）已保存为 'prediction_scatter_status.pdf'")
plt.show()
