import os
import pandas as pd

# 获取文件路径
file_path = "backtest/tick/BTCUSDT/parquet/BTCUSDT_5m_2024-12-01_2024-12-31_chunk_1.parquet"

# 读取Parquet文件
df = pd.read_parquet(file_path)

# 获取文件名
file_name = os.path.basename(file_path)

# 提取文件名中的最后一个数字
last_number = int(file_name.split("_")[-1].split(".")[0])

# 执行-1操作
new_number = last_number - 1

# 生成新的文件名
new_file_name = file_name.replace(str(last_number), str(new_number))

# 生成新的文件路径
new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

# 保存新的Parquet文件
df.to_parquet(new_file_path)

# 删除原始文件
os.remove(file_path)