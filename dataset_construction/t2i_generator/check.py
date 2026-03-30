import os
import pyarrow.parquet as pq
import pandas as pd

# 随便挑一个刚才生成的 parquet 文件
file_path = "/fs/projects/SGH_CR_RAI-AP_szh-hpc_users/workplace/mot2sgh/ImageAttributionBench-hf/data/ideogram/AnimalFace/wild.parquet"

def inspect_parquet(path):
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    print("="*40)
    print(f"1. 文件基础信息")
    print("="*40)
    file_size = os.path.getsize(path) / (1024*1024)
    print(f"File Size: {file_size:.4f} MB")
    
    pq_file = pq.ParquetFile(path)
    print(f"Num Rows: {pq_file.metadata.num_rows}")
    
    print("\n" + "="*40)
    print(f"2. Parquet Schema (底层表结构)")
    print("="*40)
    print(pq_file.schema)

    print("\n" + "="*40)
    print(f"3. 第一行数据内容深度解剖")
    print("="*40)
    df = pd.read_parquet(path)
    first_row = df.iloc[0]
    
    for col_name in first_row.keys():
        val = first_row[col_name]
        print(f"\n[{col_name}] -> Type: {type(val)}")
        
        # 重点解剖 image 列
        if col_name == 'image':
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, bytes):
                        print(f"   ├── {k}: <bytes object, length={len(v)}>")
                    else:
                        print(f"   ├── {k}: {v}")
            else:
                print(f"   └── Content: {val}")
        else:
            print(f"   └── Content: {val}")

inspect_parquet(file_path)