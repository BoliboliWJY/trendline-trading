# %%
"""
整理后的 main 模块

此模块根据配置文件判断是实盘交易模式还是回测模式，
加载相应数据，并生成对应的趋势数据，最后继续后续操作或可视化展示。
"""

import time
import datetime
import numpy as np
from binance.spot import Spot as Client

# from binance import Client
from src.config_manager import load_basic_config
from src.data_loader import load_or_fetch_data
from src.trend_calculator.trend_generator import backtest_calculate_trend_generator
from src.trend_calculator.trend_process import initial_single_slope, calculate_trend
from src.filter.filters import filter_trend
from src.time_number import time_number
from src.latest_data.new_price import get_current_price
from src.latest_data.latest_klines import get_latest_klines
from src.backtester.backtester import Backtester
from src.utils import profile_method
from src.plotter.plotter import Plotter

from src.get_data.data_getter import data_getter

# 回测tick价格管理器
from src.get_data.backtest_tick_price_getter import BacktestTickPriceManager

# 交易员
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
    client = Client(key, secret)

    # --------------------------
    # 分情况处理：实盘交易模式 / 回测模式
    # --------------------------
    if run_type:
        # ---------------------
        # 实盘交易模式
        # ---------------------
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
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
        )
        backtest_end_time = int(
            datetime.datetime.strptime(backtest_end_time_str, "%Y-%m-%d")
            .replace(tzinfo=datetime.timezone.utc)
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
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
            * 1000
        )
        index = np.searchsorted(data[:, 5], backtest_calculate_time, side="left")
        index = index - 1
        if index < len(data):
            print(
                f"第一个满足 data[:,0] > aim_time 的索引是 {index}, 时间戳为 {data[index, 0]}"
            )
        else:
            print("未找到满足 data[:,0] > aim_time 的数据点")

        visualize_mode = basic_config["visualize_mode"]
        visual_number = basic_config["visual_number"]
        base_trend_number = index  # 可根据需要固定，也可以用 index

        # 5. 初始化趋势生成器与回测器
        trend_generator = backtest_calculate_trend_generator(
            data=data,
            initial_single_slope=initial_single_slope,
            calculate_trend=calculate_trend,
        )

        backtester = Backtester(
            data,
            type_data,
            trend_generator,
            filter_trend,
            trend_config,
            base_trend_number,
        )
        initial_trend_data = backtester.initial_trend_data
        current_trend_high = initial_trend_data["trend_high"]
        current_trend_low = initial_trend_data["trend_low"]

        backtest_trader = Trader()

        # 6. 根据配置决定是否可视化
        if visualize_mode:
            delay = trend_config.get("delay") if trend_config.get("enable_filter", True) else 1
            cache_len = 1000  # 缓存长度
            plotter = Plotter(
                data, type_data, initial_trend_data, visual_number, delay, cache_len
            )
            for current_trend in backtester.run_backtest():
                # 如果处于暂停状态，则持续等待
                while plotter.paused:
                    plotter.run()
                    time.sleep(0.05)
                plotter.update_plot(current_trend)
                plotter.run()
            print("回测可视化已结束，继续后续操作...")
        else:

            trend_count = 1
            filter_count = 0

            for current_trend in backtester.run_backtest():
                # 保存上一次的趋势

                last_trend_high = current_trend_high
                last_trend_low = current_trend_low
                if current_trend["removing_item"] == True:  # 如果趋势被过滤
                    # 载入趋势数据
                    backtest_trader.get_trend_data(
                        data, base_trend_number, last_trend_high, last_trend_low
                    )
                    lower_bound = data[base_trend_number - 1, 6]
                    upper_bound = lower_bound + trend_config["interval"]
                    for p in backtest_tick_price.yield_prices_from_filtered_data(
                        lower_bound, upper_bound
                    ):
                        pass
                        # print(p)

                    filter_count += 1
                else:  # 如果趋势未被过滤，则更新当前趋势
                    current_trend_high = current_trend["trend_high"]
                    current_trend_low = current_trend["trend_low"]

                trend_count += 1
                base_trend_number += 1
            print("处理的趋势数量:", trend_count)
            print("通过过滤器处理的趋势数量:", filter_count)
            print("通过过滤器处理的趋势比例:", filter_count / trend_count)
            print("回测无可视化模式下已结束，继续后续操作...")

        backtest_elapsed_time = time.perf_counter() - start_time
        print(f"回测耗时: {backtest_elapsed_time:.6f} 秒")


if __name__ == "__main__":
    main()
