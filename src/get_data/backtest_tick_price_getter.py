# %%
from datetime import datetime
import os
import zipfile
import polars as pl
import numpy as np
import tqdm
import time
from datetime import timedelta

class BacktestTickPriceManager:
    def __init__(
        self,
        coin_type: str,
        backtest_start_time: str,
        backtest_end_time: str,
        contract_type: str,
        time_interval,
    ):
        self.coin_type = coin_type
        self.backtest_start_time = backtest_start_time
        self.backtest_end_time = backtest_end_time
        self.contract_type = contract_type
        self.time_interval = time_interval

        self.start_date = datetime.strptime(backtest_start_time, "%Y-%m-%d")
        self.end_date = datetime.strptime(backtest_end_time, "%Y-%m-%d")
        # self.tick_price = self.get_tick_price()
        
        self.current_load_date = None
        self.chunk_counter = 0

    def package_data(self, start_time_number, end_time_number, base_filename):
        """
        将时间戳和价格数据打包成一个数组，
        如果检测到 time 数据的单位为 microseconds 则转换为 milliseconds。
        """
        folder = os.path.dirname(base_filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        
        # 计算每天应该有的chunk数量
        expected_chunks_per_day = 24 * 60 * 60 // self.time_interval
        
        # 检查每天文件是否完整
        current_date = self.start_date
        incomplete_days = []
        
        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            day_folder = f"{base_filename}/{date_str}"
            
            # 检查文件夹是否存在
            if not os.path.exists(day_folder):
                incomplete_days.append(current_date)
                print(f"日期 {date_str} 的文件夹不存在")
            else:
                # 检查实际chunk文件数量
                actual_chunks = len([f for f in os.listdir(day_folder) if f.startswith("chunk_") and f.endswith(".parquet")])
                if actual_chunks != expected_chunks_per_day:
                    incomplete_days.append(current_date)
                    print(f"日期 {date_str} 的文件数量不一致: 期望 {expected_chunks_per_day}, 实际 {actual_chunks}")
            
            current_date = current_date + timedelta(days=1)
        
        if not incomplete_days:
            print(f"所有日期的文件数量都正确，跳过读取和打包。")
            return
        
        # 只有当有不完整的日期时才读取csv数据
        print(f"需要处理的日期: {[d.strftime('%Y-%m-%d') for d in incomplete_days]}")
        self.tick_price = self.get_tick_price()

        # 检查 time 字段的最大值是否异常大，
        # 与当前时间的毫秒数进行比较，判断是否为 microseconds 单位
        current_time_ms = int(time.time() * 1000)
        max_time = self.combined_df["time"].max()
        if max_time > current_time_ms * 2:  # 如果 max_time 明显比当前毫秒时间大，视为 microseconds
            print("检测到 time 字段单位为 microseconds，转换为 milliseconds")
            self.combined_df = self.combined_df.with_columns((pl.col("time") / 1000).alias("time"))

        time_series = self.combined_df["time"]
        
        # 只处理不完整的日期
        # HACK: 这里记得注释掉
        incomplete_days = []
        for current_date in incomplete_days:
            date_str = current_date.strftime("%Y-%m-%d")
            day_start = int(datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
            day_end = int(datetime.strptime(f"{date_str} 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
            
            print(f"处理日期: {date_str}")
            
            # 清理当前日期下的所有chunk文件
            day_folder = f"{base_filename}/{date_str}"
            if os.path.exists(day_folder):
                for f in os.listdir(day_folder):
                    if f.startswith("chunk_") and f.endswith(".parquet"):
                        os.remove(os.path.join(day_folder, f))
            
            # 为每一天重置chunk计数器
            chunk_counter = 0
            os.makedirs(day_folder, exist_ok=True)
            
            for i, (start_time, end_time) in enumerate(zip(start_time_number, end_time_number)):
                if start_time >= day_start and start_time < day_end:
                    filename = f"{base_filename}/{date_str}/chunk_{chunk_counter}.parquet"
                    filename = f"{base_filename}/{date_str}/chunk_{chunk_counter}.parquet"
                    # 获取时间段数据
                    start_idx = time_series.search_sorted(start_time, side="right")
                    end_idx = time_series.search_sorted(min(end_time, day_end), side="left")
                    
                    if start_idx < end_idx:
                        df_chunk = self.combined_df.slice(start_idx, end_idx - start_idx)
                        df_chunk.write_parquet(filename)
                        chunk_counter += 1  # 只有成功写入文件后才增加计数器

    
    def get_tick_price(self):
        self.dfs = []
        self.base_folder = os.path.join("backtest", self.contract_type, "tick", self.coin_type)
        
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            zip_file_name = f"{self.coin_type}-trades-{date_str}.zip"
            csv_file_name = f"{self.coin_type}-trades-{date_str}.csv"

            zip_file_path = os.path.join(self.base_folder, zip_file_name)
            if os.path.exists(zip_file_path):
                try:
                    with zipfile.ZipFile(zip_file_path, "r") as zf:
                        with zf.open(csv_file_name) as f:
                            df = pl.read_csv(f)
                            if "time" not in df.columns:
                                cols = list(df.columns)
                                if len(cols) >= 5:
                                    df = df.rename({
                                        cols[0]: "id",
                                        cols[1]: "price",
                                        cols[2]: "qty",
                                        cols[3]: "quote_qty",
                                        cols[4]: "time",
                                        cols[5]: "is_buyer_maker"
                                    })
                                else:
                                    print(f"压缩包 {zip_file_path} 中的 {csv_file_name} 不含足够列进行重命名.")
                                    current_date += timedelta(days=1)
                                    continue
                            self.dfs.append(df)
                except Exception as e:
                    print(f"无法从压缩包 {zip_file_path} 读取文件 {csv_file_name}: {e}")
            else:
                print(f"未找到文件 {zip_file_path}")
            
            current_date = current_date + timedelta(days=1)
        
        if not self.dfs:
            raise Exception("No data found for the given time range.")
        
        self.combined_df = pl.concat(self.dfs, rechunk = True)
        
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

    def package_data_loader(self, base_filename: str, timestamp: int):
        """
        读取 package_data 打包的数据，并过滤掉相邻时间点中价格相同的记录
        
        Args:
            base_filename: 基础文件名
            timestamp: 时间戳（毫秒），用于确定具体日期
        """
        try:
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
            # 如果日期发生变化，则重置chunk计数器
            if date_str != self.current_load_date:
                self.current_load_date = date_str
                self.chunk_counter = 0
            
            filename = f"{base_filename}/{date_str}/chunk_{self.chunk_counter}.parquet"
            filename = f"{base_filename}/{date_str}/chunk_{self.chunk_counter}.parquet"
            if not os.path.exists(filename):
                print(f"Error: 文件 {filename} 未找到。")
                return None
            
            # 先读取数据并选取 "price" 和 "time" 列
            df = pl.read_parquet(filename).select(["time", "price"])
            data = df.to_numpy()

            # 如果数据量不为0，则构造布尔掩码来过滤相邻价格相同的记录
            if data.shape[0] > 0:
                # 第一个数据点保留；后续数据点与前一个数据点的价格不同则保留
                mask = np.concatenate(([True], data[1:, 1] != data[:-1, 1]))
                filtered_data = data[mask]
            else:
                filtered_data = data

            # 成功读取后增加计数器
            self.chunk_counter += 1
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
        contract_type,
    ):
        """
        修改数据。在执行后续修改前，先检查是否已处理过数据，
        只要第一组数据的最大价格对应的时间能够对上，就认为数据已修改过。
        """
        if data is None or data.shape[0] == 0:
            print(f"Error: 数据为空")
            return data

        # 检查是否已处理
        first_time = data[base_trend_number, 0]
        first_date = datetime.fromtimestamp(first_time / 1000).strftime("%Y-%m-%d")
        
        try:
            first_chunk_filename = f"{base_filename}/{first_date}/chunk_0.parquet"
            print(f"尝试读取文件: {first_chunk_filename}")
            if os.path.exists(first_chunk_filename):
                print(f"文件存在: {first_chunk_filename}")
                df_check = pl.read_parquet(first_chunk_filename)
                arr_check = df_check.select(["price", "time"]).to_numpy()
                if arr_check.shape[0] > 0:
                    first_max_index = np.argmax(arr_check[:, 0])
                    first_max_time = arr_check[first_max_index, 1]
                    if data[base_trend_number, 0] == first_max_time:
                        print("数据已经处理过，跳过修改。")
                        return data
                else:
                    print(f"Warning: 文件 {first_chunk_filename} 为空，无法验证是否已处理")
            else:
                print(f"文件不存在: {first_chunk_filename}")
        except Exception as e:
            print(f"检查数据是否已处理时发生错误: {e}")

        # 使用字典记录每个日期的chunk计数
        daily_chunk_counters = {}
        
        with tqdm.tqdm(
            total=data.shape[0] - base_trend_number, desc="Processing data"
        ) as pbar:
            current_date = None
            chunk_counter = 0
            
            for i in range(data.shape[0] - base_trend_number):
                current_time = data[i + base_trend_number, 0]
                new_date = datetime.fromtimestamp(current_time / 1000).strftime("%Y-%m-%d")
                
                # 如果日期变化，重置计数器
                if new_date != current_date:
                    current_date = new_date
                    chunk_counter = 0
                    if current_date not in daily_chunk_counters:
                        daily_chunk_counters[current_date] = 0
                
                filename = f"{base_filename}/{current_date}/chunk_{chunk_counter}.parquet"
                filename = f"{base_filename}/{current_date}/chunk_{chunk_counter}.parquet"
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
                
                chunk_counter += 1
                pbar.update(1)

        self.save_modified_data(
            data, coin_type, interval, length, backtest_start_time, backtest_end_time, contract_type
        )

        return data

    def save_modified_data(
        self, data, coin_type, interval, length, backtest_start_time, backtest_end_time, contract_type
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
        from src.get_data import get_backtest_filename

        # 根据 coin_type、interval 等参数获取保存的路径和文件名
        filename, typename = get_backtest_filename(
            coin_type, interval, length, backtest_start_time, backtest_end_time, contract_type
        )

        try:
            np.save(filename, data)
            print(f"数据已成功保存至：{filename}")
        except Exception as e:
            print(f"保存数据到 {filename} 时发生错误: {e}")
