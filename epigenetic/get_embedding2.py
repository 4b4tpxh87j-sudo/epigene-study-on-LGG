import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np
try:
    merged_data = pd.read_hdf(
        '/home/luhc/epigenetic/epigenetic/lgg_merged_data', 
        key='lgg_merged_data' 
    )
except FileNotFoundError:
    print("错误：未找到文件。请检查路径 '/home/luhc/epigenetic/aligned_LGG_data.h5' 是否正确。")

print(merged_data.head())

X=merged_data.iloc[:, :-2]
Y=merged_data.iloc[:,-2:]
X_dropped_nan=X.dropna(axis=1,how='all')
removed_cols=X.shape[1]-X_dropped_nan.shape[1]
if removed_cols > 0:
    print(f"\n--- NaN 列处理 ---")
    print(f"已移除 {removed_cols} 列（这些列的所有值都是 NaN）。")
    print(f"剩余特征 X 形状: {X_dropped_nan.shape}")
    X = X_dropped_nan
else:
    print(f"\n--- NaN 列处理 ---")
    print(f"没有发现所有值都是 NaN 的特征列。")
#其他的NAN值则不进行填充，见于甲基化的相互之间弱相关的性质
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

#数据筛选
variance_threshold=0.01
selector=VarianceThreshold(threshold=variance_threshold)
selector.fit(X_train)
X_train_filtered=X_train.loc[:,selector.get_support()]
X_test_filtered = X_test.loc[:, selector.get_support()]
print("\n--- 低方差过滤完成 ---")
print(f"过滤前探针数量: {X.shape[1]}")
print(f"过滤后探针数量: {X_train_filtered.shape[1]}")
print(f"移除的探针数量: {X.shape[1] - X_train_filtered.shape[1]}")

print("\n最终训练集特征形状 (X_train_filtered):", X_train_filtered.shape)
print("最终测试集特征形状 (X_test_filtered):", X_test_filtered.shape)

# 假设您在之前的步骤中已经定义了 X_train_filtered, X_test_filtered, Y_train, Y_test
# 我们将它们保存在一个新的 HDF5 文件中，方便后续读取

output_h5_path = '/home/luhc/epigenetic/epigenetic/filtered_LGG_train_test_data.h5'

#以写入模式 ('w') 打开文件，如果文件存在则覆盖
with pd.HDFStore(output_h5_path, mode='w') as store:
    store.put('X_train_filtered', X_train_filtered, format='fixed')
    store.put('X_test_filtered', X_test_filtered, format='fixed')
    store.put('Y_train', Y_train, format='fixed')
    store.put('Y_test', Y_test, format='fixed')

print("\n--- 保存完成 ---")
print(f"所有训练集和测试集数据已成功保存到文件：{output_h5_path}")
print("您可以通过不同的 key ('X_train_filtered', 'Y_test' 等) 来读取它们。")