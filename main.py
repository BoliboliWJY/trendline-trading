# %%
"""
整理后的 main 模块

此模块根据配置文件判断是实盘交易模式还是回测模式，
加载相应数据，并生成对应的趋势数据，最后继续后续操作或可视化展示。
"""

import time
from tqdm import tqdm
import datetime
import numpy as np
import copy

# from binance.spot import Spot as Client
from binance.um_futures import UMFutures  # 导入U本位合约市场客户端


# from binance import Client
from src.config_manager import load_basic_config
from src.trend_calculator.trend_generator import backtest_calculate_trend_generator
from src.trend_calculator.trend_process import initial_single_slope, calculate_trend
from src.filter.filters import trend_filter
from src.time_number import time_number
from src.latest_data.new_price import get_current_price
from src.latest_data.latest_klines import get_latest_klines
from src.backtester.backtester import Backtester
from src.utils import profile_method
from src.plotter.plotter import Plotter

from src.get_data.data_getter import data_getter

# 回测tick价格管理器
from src.get_data.backtest_tick_price_getter import BacktestTickPriceManager

# 趋势价格计算器
from src.backtester.trend_tick_calculator import TrendTickCalculator

from src.trader.trader import Trader

# %%
def main():
    """
    根据配置文件加载参数，并区分实盘交易或回测模式进行处理。
    """
    # --------------------------
    # 公共部分：加载配置、创建客户端
    # --------------------------
    # 记录开始时间
    start_time = time.perf_counter()

    # 加载基本配置
    basic_config = load_basic_config("config/basic_config.yaml")
    key = basic_config["key"]
    secret = basic_config["secret"]
    coin_type = basic_config["coin_type"]
    contract_type = basic_config["contract_type"]
    # aim_time_str = basic_config["aim_time"]
    # total_length = basic_config["total_length"]
    interval = basic_config["interval"]
    run_type = basic_config["run_type"]  # True: 实盘；False: 回测

    # 趋势参数
    trend_config = basic_config["trend_config"]
    trading_config = basic_config["trading_config"]
    # 将趋势间隔时间转换为毫秒单位
    trend_config["interval"] = time_number(trend_config["interval"]) * 1000

    # 创建 Binance 客户端
    client = UMFutures(key, secret)

    # --------------------------
    # 分情况处理：实盘交易模式 / 回测模式
    # --------------------------
    if run_type:
        # ---------------------
        # 实盘交易模式
        # ---------------------
        return
        current_price = get_current_price(client, coin_type)
        print("当前价格:", current_price)
        data, type_data = get_latest_klines(client, coin_type, interval, total_length)
        print("最新 K 线数据:", data[-1])

        # 初始化趋势生成器
        trend_generator = backtest_calculate_trend_generator(
            data=data,
            initial_single_slope=initial_single_slope,
            calculate_trend=calculate_trend,
        )

        # 如果需要继续添加其它实盘操作，可在此后添加
    else:
        # ---------------------
        # 回测模式
        # ---------------------
        # 1. 读取回测所需的时间区间，并转换为时间戳
        backtest_config = basic_config["backtest_config"]
        backtest_start_time_str = backtest_config["backtest_start_time"]
        backtest_end_time_str = backtest_config["backtest_end_time"]
        backtest_calculate_time_str = backtest_config["backtest_calculate_time"]

        # 获取历史tick数据（回测时段的数据）
        backtest_tick_price = BacktestTickPriceManager(
            coin_type, backtest_calculate_time_str, backtest_end_time_str
        )

        backtest_start_time = int(
            datetime.datetime.strptime(backtest_start_time_str, "%Y-%m-%d")
            .replace(
                tzinfo=datetime.timezone(datetime.timedelta(hours=8))
            )  # 使用 UTC+8 时区
            .timestamp()
        )
        backtest_end_time = int(
            datetime.datetime.strptime(backtest_end_time_str, "%Y-%m-%d")
            .replace(
                tzinfo=datetime.timezone(datetime.timedelta(hours=0))
            )  # 使用 UTC+0 时区
            .timestamp()
        )

        # 2. 计算数据长度（这里将时间间隔转化为秒数进行整除）
        length = int((backtest_end_time - backtest_start_time) // time_number(interval))

        # 3. 获取历史k线数据（回测时段的数据）
        data, type_data = data_getter(
            client, coin_type, interval, length, backtest_start_time, backtest_end_time
        )

        elapsed_time = time.perf_counter() - start_time
        print(f"获取数据耗时: {elapsed_time:.6f} 秒")

        # 4. 计算回测起始计算点的索引（转换到毫秒单位后查找对应数据位置）
        backtest_calculate_time = (
            int(
                datetime.datetime.strptime(backtest_calculate_time_str, "%Y-%m-%d")
                .replace(
                    tzinfo=datetime.timezone(datetime.timedelta(hours=0))
                )  # 使用 UTC+0 时区
                .timestamp()
            )
            * 1000
        )
        index = np.searchsorted(data[:, 6], backtest_calculate_time, side="left") - 1 # 搜索开盘时间前一个点
        if index < len(data):
            print(
                f"第一个满足 data[:,0] > aim_time 的索引是 {index}, 时间戳为 {data[index, 0]}"
            )
        else:
            print("未找到满足 data[:,0] > aim_time 的数据点")

        visualize_mode = basic_config["visualize_mode"]
        visual_number = basic_config["visual_number"]
        base_trend_number = index + 1  # 可根据需要固定，也可以用 index

        parquet_filename = f"backtest/tick/BTCUSDT/parquet/{coin_type}_{interval}_{backtest_calculate_time_str}_{backtest_end_time_str}"
        backtest_tick_price.package_data(
            data[base_trend_number:, 6], data[base_trend_number:, 7], parquet_filename
        )
        data = backtest_tick_price.modify_data(
            data,
            base_trend_number,
            parquet_filename,
            coin_type,
            interval,
            length,
            backtest_start_time,
            backtest_end_time,
        )
        parquet_index = 1

        # 初始化回测器
        backtester = Backtester(data, type_data, base_trend_number)
        initial_trend_data = backtester.initial_trend_data
        # 过滤趋势
        filter_trend = trend_filter(data,trend_config)
        # 初始过滤趋势数据
        initial_filtered_trend_data = filter_trend.filter_trend_initial(
            initial_trend_data["trend_high"],
            initial_trend_data["trend_low"],
        )  
        
        # 初始化趋势价格计算器
        trend_tick_calculator = TrendTickCalculator(data, trend_config, initial_filtered_trend_data)
        
        # # 初始化交易器
        trader = Trader(data, trend_config, trading_config)

        # 6. 根据配置决定是否可视化
        if visualize_mode:
            cache_len = 1000  # 缓存长度
            if base_trend_number < visual_number:
                error_msg = f"base_trend_number 小于 visual_number，请检查配置文件"
                raise ValueError(error_msg)
            plotter = Plotter(
                data,
                type_data,
                initial_filtered_trend_data,
                base_trend_number,
                visual_number,
                cache_len,
            )
            # 初始化交易器
            # trader = Trader(data, trend_config, trading_config, plotter)
            
            base_trend_number += 1
            # start_time = time.time()
            # backtester.run_backtest()
            # backtester.run_backtest()
            
            idx_draw = 0
            
            with tqdm(desc="Backtesting Progress", mininterval=1) as pbar:
                for current_trend in backtester.run_backtest():
                    while plotter.paused:
                        plotter.run()
                        time.sleep(0.05)
                    
                    # 过滤趋势
                    filtered_trend_data = filter_trend.process_new_trend(
                        initial_filtered_trend_data,
                        current_trend,
                    )
                    
                    # 计算趋势价格
                    trend_tick_data = trend_tick_calculator.update_trend_data(data, base_trend_number, filtered_trend_data)
                    
                    # 获取价格时间数组
                    price_time_array = backtest_tick_price.package_data_loader(
                        parquet_index, parquet_filename
                    )
                    parquet_index += 1
                    
                    trader.open_close_signal(data[base_trend_number,[1,3,4,5]],trend_tick_data, price_time_array)
                    
                    
                    if trader.paused:
                        plotter.enable_visualization = True
                    else:
                        plotter.enable_visualization = False
                    open_signals = trader.open_signals
                    close_signals = trader.close_signals
                    
                    # plotter.enable_visualization = True
                    
                    # price_time_array = np.array([])
                    
                    plotter.update_plot(
                        filtered_trend_data,
                        open_signals,
                        close_signals,
                        base_trend_number,
                        price_time_array,
                        trend_tick_data,
                    )
                    
                    plotter.run()
                    
                    if plotter.enable_visualization:
                        plotter.save_frame(coin_type)

                    base_trend_number += 1


                    pbar.update(1)

            print("回测可视化已结束，继续后续操作...")
        
        
        else:
            # return
            base_trend_number += 1
            # start_time = time.time()
            with tqdm(desc="Backtesting Progress", mininterval=1) as pbar:
                for current_trend in backtester.run_backtest():
                    
                    # 过滤趋势
                    filtered_trend_data = filter_trend.process_new_trend(
                        initial_filtered_trend_data,
                        current_trend,
                    )
                    
                    # 计算趋势价格
                    trend_tick_data = trend_tick_calculator.update_trend_data(data, base_trend_number, filtered_trend_data)
                    
                    # 获取价格时间数组
                    price_time_array = backtest_tick_price.package_data_loader(
                        parquet_index, parquet_filename
                    )
                    parquet_index += 1
                    
                    trader.open_close_signal(data[base_trend_number,[1,3]],trend_tick_data, price_time_array)
                    
                    


                    base_trend_number += 1


                    pbar.update(1)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", sort="cumulative")
    main()
    