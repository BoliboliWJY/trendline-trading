import polars as pl
import time

s_time = time.time()

df = pl.read_csv("BTCUSDT-trades-2024-12.csv")

# 获取某一列数据
price_series = df["price"]
time_series = df["time"]
# 转换为 Python 列表或 NumPy 数组进行进一步处理（通常这种操作放在特定场景下使用）
# 或者
price_np = price_series.to_numpy()
time_np = time_series.to_numpy()
length = len(price_np)
print("length:", length)
print("price_np[0]:", price_np[0])
print("price_np[-1]:", price_np[-1])
print("time_np[0]:", time_np[0])
print("time_np[-1]:", time_np[-1])
e_time = time.time()
print("Polars in eager mode, time taken:", (e_time - s_time), "sec")
