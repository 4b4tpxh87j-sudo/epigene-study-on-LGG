import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 加载与过滤 (保持之前的死亡样本过滤逻辑)
h5_path = '/home/luhc/epigenetic/epigenetic/filtered_LGG_train_test_data.h5'
X_train = pd.read_hdf(h5_path, key='X_train_filtered')
X_test = pd.read_hdf(h5_path, key='X_test_filtered')
Y_train = pd.read_hdf(h5_path, key='Y_train')
Y_test = pd.read_hdf(h5_path, key='Y_test')

train_mask = Y_train.iloc[:, 1] == 1
test_mask = Y_test.iloc[:, 1] == 1
X_train, Y_train = X_train[train_mask], Y_train[train_mask]
X_test, Y_test = X_test[test_mask], Y_test[test_mask]

# 2. 预处理
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test[X_train.columns]
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

y_train_time = Y_train.iloc[:, 0].values
y_test_time = Y_test.iloc[:, 0].values

# 3. 有监督初筛 (k=2000)
selector = SelectKBest(score_func=f_regression, k=2000)
X_train_sub = selector.fit_transform(X_train_sc, y_train_time)
X_test_sub = selector.transform(X_test_sc)

# 4. Lasso 二次筛选 (得到那 25 个位点)
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_sub, y_train_time)

# 提取这 25 个特征的索引和名称
selected_indices = np.where(lasso.coef_ != 0)[0]
mask_kbest = selector.get_support()
feature_names_all = X_train.columns
feature_names_kbest = feature_names_all[mask_kbest]
final_feature_names = feature_names_kbest[selected_indices]

print(f"\n--- 特征挖掘结果 ---")
print(f"Lasso 选出的核心位点数量: {len(final_feature_names)}")
print(f"位点名称清单: {final_feature_names.tolist()}")

# 5. 非线性模型尝试：随机森林 (只用这 25 个位点)
X_train_final = X_train_sub[:, selected_indices]
X_test_final = X_test_sub[:, selected_indices]

rf_final = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
rf_final.fit(X_train_final, y_train_time)
Y_pred_rf = rf_final.predict(X_test_final)

# 6. 评估结果
r2_rf = r2_score(y_test_time, Y_pred_rf)
mae_rf = mean_absolute_error(y_test_time, Y_pred_rf)

print(f"\n--- 最终模型评估 (RF on Lasso Features) ---")
print(f"平均绝对误差 (MAE): {mae_rf:.2f} 天")
print(f"R² 分数: {r2_rf:.4f}")

# 7. 可视化：真实 vs 预测
plt.figure(figsize=(10, 6))
plt.scatter(y_test_time, Y_pred_rf, alpha=0.7, c='teal', edgecolors='w', label='RF Predictions')
plt.plot([y_test_time.min(), y_test_time.max()], [y_test_time.min(), y_test_time.max()], 
         'r--', lw=2, label='Ideal Alignment')

# 标注位点贡献度 (Feature Importance)
plt.title(f'Final Model Performance (R²={r2_rf:.4f})\nTop 25 Methylation Sites Selected')
plt.xlabel('Actual Survival Days')
plt.ylabel('Predicted Survival Days')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('final_optimized_prediction.pdf')
plt.show()

# 8. 绘制特征重要性图
plt.figure(figsize=(10, 8))
importances = rf_final.feature_importances_
sorted_idx = np.argsort(importances)
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='skyblue')
plt.yticks(range(len(sorted_idx)), [final_feature_names[i] for i in sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.title('Importance Score of Top 25 Methylation Sites')
plt.tight_layout()
plt.savefig('feature_importance.pdf')
plt.show()