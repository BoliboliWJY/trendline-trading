import os


def batch_rename_files():
    """
    批量将目标文件夹内的所有文件名中的字符串
    "2024-11-30_2024-12-31_2024-12-01"
    替换为 "2024-12-01_2024-12-31"。
    例如：
      BTCUSDT_3m_2024-11-30_2024-12-31_2024-12-01_chunk_0.parquet
    将被重命名为：
      BTCUSDT_3m_2024-12-01_2024-12-31_chunk_0.parquet
    """
    # 指定目标文件夹，注意：路径格式根据系统（Windows 下的反斜杠）设置
    target_folder = r"backtest\tick\BTCUSDT\parquet"

    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(target_folder):
        # 这里仅示例处理含有 "BTCUSDT_3m_" 且后缀为 .parquet 的文件
        if filename.endswith(".parquet") and "BTCUSDT_3m_" in filename:
            # 重命名操作：使用字符串替换实现需要的修改
            new_filename = filename.replace(
                "2024-11-30_2024-12-31_2024-12-01", "2024-12-01_2024-12-31"
            )
            # 如果文件名发生变化，则执行重命名
            if new_filename != filename:
                original_path = os.path.join(target_folder, filename)
                new_path = os.path.join(target_folder, new_filename)
                os.rename(original_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    batch_rename_files()
