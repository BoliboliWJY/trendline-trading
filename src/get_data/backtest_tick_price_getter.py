# %%
from datetime import datetime
import os
import zipfile
import polars as pl
import numpy as np
import tqdm

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
        # self.tick_price = self.get_tick_price()

    def package_data(self, start_time_number, end_time_number, base_filename):
        """
        将时间戳和价格数据打包成一个数组
        """
        # 先检查文件是否已存在，假设只检查第一个 chunk 的存在性
        first_chunk_file = f"{base_filename}_chunk_0.parquet"
        if os.path.exists(first_chunk_file):
            print(f"文件 {first_chunk_file} 已存在，跳过读取和打包。")
            return

        # 如果未检测到文件，则读取 csv 文件后进行后续操作
        self.tick_price = self.get_tick_price()
        time_series = self.combined_df["time"]
        for i, (start_time, end_time) in enumerate(
            zip(start_time_number, end_time_number)
        ):
            # 构建输出文件名
            filename = f"{base_filename}_chunk_{i}.parquet"
            start_idx = time_series.search_sorted(start_time, side="right")
            end_idx = time_series.search_sorted(end_time, side="left")
            df_chunk = self.combined_df.slice(start_idx, end_idx - start_idx)
            df_chunk.write_parquet(filename)

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

    def yield_prices_from_filtered_data(self, lower_bound: int, upper_bound: int):
        """
        调用 filter_prices_sorted_optimized 获取数据子表后，
        逐个 yield 子表中 price 列的值。
        """
        filtered_df = self.filter_prices_sorted_optimized(lower_bound, upper_bound)
        # 假设子表中存在 "price" 这一列
        # for price, timestamp in zip(filtered_df["price"], filtered_df["time"]):
        #     yield price, timestamp

        price_time_array = filtered_df.select(["price", "time"]).to_numpy()
        return price_time_array

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

    def package_data_loader(self, chunk_index: int, base_filename: str):
        """
        读取 package_data 打包的数据，并过滤掉相邻时间点中价格相同的记录
        """
        filename = f"{base_filename}_chunk_{chunk_index}.parquet"
        try:
            # 先读取数据并选取 "price" 和 "time" 列
            df = pl.read_parquet(filename).select(["price", "time"])
            data = df.to_numpy()

            # 如果数据量不为0，则构造布尔掩码来过滤相邻价格相同的记录
            if data.shape[0] > 0:
                # 第一个数据点保留；后续数据点与前一个数据点的价格不同则保留
                mask = np.concatenate(([True], data[1:, 0] != data[:-1, 0]))
                filtered_data = data[mask]
            else:
                filtered_data = data

            return filtered_data
        except FileNotFoundError:
            print(f"Error: 文件 {filename} 未找到。")
            return None
        except Exception as e:
            print(f"Error: 读取文件时发生错误: {e}")
            return None

    def modify_data(
        self,
        data,
        base_trend_number,
        base_filename,
        coin_type,
        interval,
        length,
        backtest_start_time,
        backtest_end_time,
    ):
        """
        修改数据。在执行后续修改前，先检查是否已处理过数据，
        只要第一组数据的最大价格对应的时间能够对上，就认为数据已修改过。
        """
        if data is None or data.shape[0] == 0:
            print(f"Error: 数据为空")
            return data

        # 检查是否已处理：读取第一个chunk的数据，计算第一组的最大价格对应的时间，
        # 如果 data 中对应位置的时间一致，则跳过后续处理。
        try:
            first_chunk_filename = f"{base_filename}_chunk_0.parquet"
            print(f"尝试读取文件: {first_chunk_filename}")  # 添加调试信息
            if os.path.exists(first_chunk_filename):  # 检查文件是否存在
                print(f"文件存在: {first_chunk_filename}")
                df_check = pl.read_parquet(first_chunk_filename)
                arr_check = df_check.select(["price", "time"]).to_numpy()
                if arr_check.shape[0] > 0:
                    first_max_index = np.argmax(arr_check[:, 0])
                    first_max_time = arr_check[first_max_index, 1]
                    # 假设第一组的数据在 data 中的索引为 base_trend_number
                    if data[base_trend_number, 0] == first_max_time:
                        print("数据已经处理过，跳过修改。")
                        return data
                else:
                    print(f"Warning: 文件 {first_chunk_filename} 为空，无法验证是否已处理")
            else:
                print(f"文件不存在: {first_chunk_filename}")  # 添加调试信息
        except Exception as e:
            print(f"检查数据是否已处理时发生错误: {e}")

        import tqdm

        with tqdm.tqdm(
            total=data.shape[0] - base_trend_number, desc="Processing data"
        ) as pbar:
            for i in tqdm.tqdm(range(data.shape[0] - base_trend_number)):
                filename = f"{base_filename}_chunk_{i}.parquet"
                try:
                    df = pl.read_parquet(filename)
                except Exception as e:
                    print(f"Error: 读取文件 {filename} 时发生错误: {e}")
                    continue

                arr = df.select(["price", "time"]).to_numpy()
                if arr.shape[0] == 0:
                    print(f"Error: 文件 {filename} 为空")
                    continue

                max_index = np.argmax(arr[:, 0])
                min_index = np.argmin(arr[:, 0])
                max_price = arr[max_index, 0]
                min_price = arr[min_index, 0]
                max_time = arr[max_index, 1]
                min_time = arr[min_index, 1]

                data[i + base_trend_number, 0] = max_time
                data[i + base_trend_number, 2] = min_time
                data[i + base_trend_number, 1] = max_price
                data[i + base_trend_number, 3] = min_price

        self.save_modified_data(
            data, coin_type, interval, length, backtest_start_time, backtest_end_time
        )

        return data

    def save_modified_data(
        self, data, coin_type, interval, length, backtest_start_time, backtest_end_time
    ):
        """
        将修改后的 data 根据 backtest_filename_getter 返回的路径模式保存到文件中

        参数:
            data: 修改后的数据数组
            coin_type: 币种，例如 "BTC"
            interval: 数据时间间隔
            length: 数据长度
            backtest_start_time: 回测开始时间
            backtest_end_time: 回测结束时间
        """
        import numpy as np
        from src.get_data.backtest_filename_getter import get_backtest_filename

        # 根据 coin_type、interval 等参数获取保存的路径和文件名
        filename, typename = get_backtest_filename(
            coin_type, interval, length, backtest_start_time, backtest_end_time
        )

        try:
            np.save(filename, data)
            print(f"数据已成功保存至：{filename}")
        except Exception as e:
            print(f"保存数据到 {filename} 时发生错误: {e}")
