import pandas as pd
import numpy as np

# 1. 加载甲基化矩阵 (通常是TSV，使用 read_csv 配合 sep='\t')
# 注意：大型文件可能需要几分钟加载，且需要大量内存
methyl_df = pd.read_csv('/home/luhc/epigenetic/epigenetic/TCGA-LGG.methylation450.tsv.gz.tsv'
, sep='\t', index_col=0,compression='gzip')

# 2. 加载临床数据
clinical_df = pd.read_csv('/home/luhc/epigenetic/epigenetic/TCGA-LGG.survival.tsv.gz.tsv', sep='\t',compression='gzip')

# 查看前几行和数据形状
meth_transposed=methyl_df.T
meth_transposed['_PATIENT']=meth_transposed.index.str[:12]
meth_transposed.set_index('_PATIENT',inplace=True)
print("形状:", meth_transposed.shape)
meth_transposed = meth_transposed[~meth_transposed.index.duplicated(keep='first')]
print("\n--- 甲基化矩阵（转置并以患者ID为索引）---")
print("形状:", meth_transposed.shape)
print(meth_transposed.head())
if 'sample' in clinical_df.columns:
    clinical_df['_PATIENT']=clinical_df['sample'].str[:12]
    clinical_df.set_index('_PATIENT',inplace=True)
    clinical_df.drop(columns=['sample'],inplace=True)
clinical_df_unique = clinical_df[~clinical_df.index.duplicated(keep='first')]    
merged_data = meth_transposed.merge(clinical_df_unique,left_index=True,right_index=True,how='inner')
print("\n--- 对齐后的最终数据 (merged_data) ---")
print("形状:", merged_data.shape)
print("现在每一行是一个患者，列包含CpG探针值和临床数据。")
print(merged_data.iloc[:, -5:].head())
merged_data.to_hdf(
    '/home/luhc/epigenetic/epigenetic/lgg_merged_data', 
    key='lgg_merged_data', # 给数据起一个名字
    mode='w'               # 'w' 表示写入 (如果文件已存在会覆盖)
)