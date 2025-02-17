# %%
from datetime import datetime
import os
import zipfile
import polars as pl
import numpy as np


class BacktestTickPriceManager:
    def __init__(
        self,
        coin_type: str,
        backtest_start_time: str,
        backtest_end_time: str,
    ):
        self.coin_type = coin_type
        self.backtest_start_time = backtest_start_time
        self.backtest_end_time = backtest_end_time

        self.start_date = datetime.strptime(backtest_start_time, "%Y-%m-%d")
        self.end_date = datetime.strptime(backtest_end_time, "%Y-%m-%d")
        self.tick_price = self.get_tick_price()

    def get_tick_price(self):
        self.months = []
        self.current_date = self.start_date.replace(day=1)
        while self.current_date <= self.end_date:
            self.months.append(self.current_date.strftime("%Y-%m"))
            if self.current_date.month == 12:
                self.current_date = self.current_date.replace(
                    year=self.current_date.year + 1, month=1
                )
            else:
                self.current_date = self.current_date.replace(
                    month=self.current_date.month + 1
                )

        self.dfs = []
        self.base_folder = os.path.join("backtest", "tick", self.coin_type)
        for month in self.months:
            csv_file_name = f"{self.coin_type}-trades-{month}.csv"
            zip_file_name = f"{self.coin_type}-trades-{month}.zip"

            zip_file_path = os.path.join(self.base_folder, zip_file_name)
            csv_file_path = os.path.join(self.base_folder, csv_file_name)

            # 优先判断csv文件是否存在
            if os.path.exists(csv_file_path):
                try:
                    df = pl.read_csv(csv_file_path)
                    self.dfs.append(df)
                except Exception as e:
                    print(f"无法读取文件 {csv_file_path}: {e}")
            # 如果csv文件不存在，则判断压缩包是否存在
            elif os.path.exists(zip_file_path):
                try:
                    with zipfile.ZipFile(zip_file_path, "r") as zf:
                        with zf.open(csv_file_name) as f:
                            df = pl.read_csv(f)
                            self.dfs.append(df)
                except Exception as e:
                    print(f"无法从压缩包 {zip_file_path} 读取文件 {csv_file_name}: {e}")
            else:
                print(f"未找到文件 {zip_file_path} 或 {csv_file_path}")

        if not self.dfs:
            raise Exception("No data found for the given time range.")

        self.combined_df = pl.concat(self.dfs, rechunk=True)
        return self.combined_df

    def filter_prices_sorted_optimized(self, lower_bound: int, upper_bound: int):
        """
        利用 Polars 内置的二分查找（search_sorted）来定位时间范围，
        避免转换为 NumPy 数组，从而进一步提升性能。

        返回:
            filtered_df: 筛选出的数据切片
            start_idx: 起始索引
            end_idx: 结束索引
        """
        time_series = self.combined_df["time"]
        # 使用 Polars 内置的二分查找，不用转为 NumPy
        start_idx = time_series.search_sorted(lower_bound, side="right")
        end_idx = time_series.search_sorted(upper_bound, side="left")

        # slice 方法的第二个参数表示切片长度
        filtered_df = self.combined_df.slice(start_idx, end_idx - start_idx)
        return filtered_df

    def yield_prices_from_filtered_data(self, lower_bound: int, upper_bound: int):
        """
        调用 filter_prices_sorted_optimized 获取数据子表后，
        逐个 yield 子表中 price 列的值。
        """
        filtered_df = self.filter_prices_sorted_optimized(lower_bound, upper_bound)
        # 假设子表中存在 "price" 这一列
        for price in filtered_df["price"]:
            yield price
