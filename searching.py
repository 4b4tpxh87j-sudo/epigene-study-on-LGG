import pandas as pd
import os

def get_cpg_annotation_final(cpg_ids, cache_file="illumina_450k_meta.csv"):
    """
    获取CpG注释的完整方案：
    1. 自动处理本地缓存
    2. 自动兼容450K官方列名
    3. 异常处理与ID校验
    """
    
    # --- 1. 数据加载部分 ---
    if os.path.exists(cache_file):
        print(f"读取本地缓存文件: {cache_file}...")
        # 读取缓存时，指定第一列为索引
        df = pd.read_csv(cache_file, low_memory=False, index_col=0)
    else:
        print("未发现本地文件，正在从 Illumina 官网下载（约200MB），请稍候...")
        url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
        try:
            # 官方CSV前7行是描述信息，需跳过；IlmnID 是唯一标识符
            df = pd.read_csv(url, low_memory=False, skiprows=7)
            
            # 清洗：移除可能的空行（Manifest末尾常有控制字符）
            df = df.dropna(subset=['IlmnID'])
            df.set_index('IlmnID', inplace=True)
            
            # 保存到本地以便下次使用
            df.to_csv(cache_file)
            print(f"下载成功并已保存至: {cache_file}")
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None

    # --- 2. 列名兼容性处理 ---
    # 定义期望的字段及其在 450K Manifest 中的真实列名
    column_mapping = {
        'CHR': 'CHR',
        'MAPINFO': 'MAPINFO',
        'UCSC_RefGene_Name': 'UCSC_RefGene_Name',
        'UCSC_RefGene_Group': 'UCSC_RefGene_Group',
        'Relation_to_Island': 'Relation_to_UCSC_CpG_Island',
        'Island_Name': 'UCSC_CpG_Islands_Name' # 修正后的450K标准列名
    }

    # 检查哪些列在当前文件中确实存在
    existing_columns = [col for col in column_mapping.values() if col in df.columns]
    
    # --- 3. 数据提取 ---
    # 过滤掉不在索引中的 ID，防止 loc 报错
    valid_ids = [idx for idx in cpg_ids if idx in df.index]
    
    if not valid_ids:
        print("错误: 输入的 CpG ID 均未在注释文件中找到。")
        return pd.DataFrame()

    result = df.loc[valid_ids, existing_columns]
    
    # 重命名列名，让输出更易读
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    result = result.rename(columns=reverse_mapping)
    
    return result

# --- 4. 测试使用 ---
if __name__ == "__main__":
    # 您可以把需要查询的 ID 放在这个列表里
    my_cpgs = ['cg00570618', 'cg02810967', 'cg03490564', 'cg03694279', 'cg04198091', 'cg05072951', 'cg07218600', 'cg07770866', 'cg08552812', 'cg09360912', 'cg10020333', 'cg11995437', 'cg12810503', 'cg13157960', 'cg15859390', 'cg16399859', 'cg17445840', 'cg18167921', 'cg18639125', 'cg19040026', 'cg20128181', 'cg20517697', 'cg20943032', 'cg21461300', 'cg23269663']
    
    annotation_data = get_cpg_annotation_final(my_cpgs)
    
    print("\n" + "="*30)
    print("CpG 位点注释查询结果")
    print("="*30)
    if not annotation_data.empty:
        print(annotation_data)
    else:
        print("无匹配结果。")