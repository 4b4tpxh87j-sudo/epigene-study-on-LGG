import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 加载数据
h5_path = '/home/luhc/epigenetic/epigenetic/filtered_LGG_train_test_data.h5'
try:
    X_train_filtered = pd.read_hdf(h5_path, key='X_train_filtered')
    X_test_filtered = pd.read_hdf(h5_path, key='X_test_filtered')
    Y_train = pd.read_hdf(h5_path, key='Y_train')
    Y_test = pd.read_hdf(h5_path, key='Y_test')
except Exception as e:
    print(f"错误：加载数据失败。错误信息：{e}")
    exit()





# 2. 缺失值处理
# 【新增：修复步骤】先剔除掉那些全是 NaN 的列，否则 Imputer 会改变矩阵形状
X_train_filtered = X_train_filtered.dropna(axis=1, how='all')
# 确保测试集也对齐训练集的列（只保留训练集里存在的列）
X_test_filtered = X_test_filtered[X_train_filtered.columns]

print(f"剔除全空列后，特征数变为: {X_train_filtered.shape[1]}")

imputer = SimpleImputer(strategy='mean')
# 注意：fit_transform 返回的是 numpy 数组
X_train_imputed = imputer.fit_transform(X_train_filtered)
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



# --- 关键修改：剔除 OS=0 (存活/截断) 的样本 ---
# 假设 Y 的第二列索引为 1，代表生存状态 (1为死亡)
train_mask = Y_train.iloc[:, 1] == 1
test_mask = Y_test.iloc[:, 1] == 1

X_train_filtered = X_train_filtered[train_mask]
Y_train = Y_train[train_mask]

X_test_filtered = X_test_filtered[test_mask]
Y_test = Y_test[test_mask]

print(f"过滤后训练集样本数 (仅死亡): {X_train_filtered.shape[0]}")
print(f"过滤后测试集样本数 (仅死亡): {X_test_filtered.shape[0]}")

# 2. 缺失值填充
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_filtered)
X_test_imputed = imputer.transform(X_test_filtered) 

X_train_filtered = pd.DataFrame(X_train_imputed, index=X_train_filtered.index, columns=X_train_filtered.columns)
X_test_filtered = pd.DataFrame(X_test_imputed, index=X_test_filtered.index, columns=X_test_filtered.columns)

# 3. PCA 降维
n_components = 100
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_filtered)
X_test_pca = pca.transform(X_test_filtered)

pca_cols = [f'PC{i+1}' for i in range(n_components)]
X_train_pca = pd.DataFrame(X_train_pca, index=X_train_filtered.index, columns=pca_cols)
X_test_pca = pd.DataFrame(X_test_pca, index=X_test_filtered.index, columns=pca_cols)

explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"前 {n_components} 个主成分总共解释了 {explained_variance*100:.2f}% 的方差。")

# 4. 随机森林回归训练 (目标为第一列：生存时间)
Y_train_time = Y_train.iloc[:, 0]
Y_test_time = Y_test.iloc[:, 0]

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_pca, Y_train_time)

# 5. 在测试集上预测
Y_pred = rf_model.predict(X_test_pca)

# 6. 计算评估指标
mse = mean_squared_error(Y_test_time, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_time, Y_pred)
r2 = r2_score(Y_test_time, Y_pred)

print("\n--- [仅针对死亡样本] 模型评估结果 ---")
print(f"平均绝对误差 (MAE): {mae:.2f} 天")
print(f"均方根误差 (RMSE): {rmse:.2f} 天")
print(f"R² 分数 (解释性评分): {r2:.4f}")

# 7. 可视化
plt.figure(figsize=(8, 8))

# 由于此时全部为 OS=1，颜色将统一，但为了代码兼容性仍保留配色逻辑
plt.scatter(Y_test_time, Y_pred, alpha=0.7, c='red', edgecolors='w', linewidth=0.5, label='Deceased (OS=1)')

# 画出 y=x 的对角参考线
all_vals = np.concatenate([Y_test_time.values, Y_pred])
max_val = all_vals.max()
min_val = all_vals.min()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Survival Time (Days)')
plt.ylabel('Predicted Survival Time (Days)')
plt.title('Actual vs Predicted Survival Time (Test Set: Deceased Only)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 保存图片
plt.savefig('prediction_scatter_deceased_only.pdf', format='pdf', bbox_inches='tight')
print("\n预测结果图已保存为 'prediction_scatter_deceased_only.pdf'")
plt.show()