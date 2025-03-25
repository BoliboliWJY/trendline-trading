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
import yaml
import gc  # 导入垃圾回收模块

# from binance.spot import Spot as Client
from binance.um_futures import UMFutures  # 导入U本位合约市场客户端
from src.coin_info import CoinInfo

# from binance import Client
from src.config_manager import load_basic_config
from src.filter.filters import trend_filter
from src.time_number import time_number
from src.latest_data.new_price import NewPrice
from src.latest_data.latest_klines import get_latest_klines
from src.backtester.backtester import Backtester
from src.utils import profile_method
from src.plotter.plotter import Plotter

from src.get_data.data_getter import data_getter
from src.get_data.tick_data_downloader import download_tick_data

# 回测tick价格管理器
from src.get_data.backtest_tick_price_getter import BacktestTickPriceManager

# 趋势价格计算器
from src.backtester.trend_tick_calculator import TrendTickCalculator

# 订单簿记录，开仓，订单管理器
from src.order_book import OrderBook, open_order, OrderManager
# 回测交易器
from src.trader.backtest_trader import BacktestTrader
# 实时交易器
from src.trader.realtime_trader import RealtimeTrader

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
    key = basic_config["api_key"]
    secret = basic_config["api_secret"]
    coin_type = basic_config["coin_type"]
    contract_type = basic_config["contract_type"]
    # aim_time_str = basic_config["aim_time"]
    # total_length = basic_config["total_length"]
    interval = basic_config["interval"]
    run_type = basic_config["run_type"]  # True: 实盘；False: 回测

    trend_config = basic_config["trend_config"]
    trading_config = basic_config["trading_config"]
    # 将趋势间隔时间转换为毫秒单位
    trend_config["interval"] = time_number(trend_config["interval"]) * 1000
    

    if basic_config["client_testnet"]:# 是不是模拟盘
        client = UMFutures(key, secret, base_url="https://testnet.binancefuture.com")
    else:
        client = UMFutures(key, secret)
        
    coin_info = CoinInfo(client, coin_type, 10)

    # --------------------------
    # 分情况处理：实盘交易模式 / 回测模式
    # --------------------------
    if run_type:
        from src.coin_info import CoinMonitorManager
        print("实盘交易模式")
        # ---------------------
        # 实盘交易模式
        # ---------------------
        client.ping() # 测试连接
        realtime_config = basic_config["realtime_config"]
        total_length = realtime_config["total_length"]
        
        coin_types = basic_config.get("coin_types", [coin_type])
        
        manager = CoinMonitorManager(client, trend_config, trading_config)
        for coin_type in coin_types:
            manager.add_coin(coin_type, contract_type, interval, total_length)
            
        manager.start_all()
        
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("手动停止")
            manager.stop_all()
            print("所有监控实例已停止")
            
        
            

        # data, type_data = get_latest_klines(client, coin_type, interval, total_length + 1)
        # # 初始化趋势生成器
        # backtester = Backtester(data, type_data)
        # initial_trend_data = backtester.initial_trend_data
        # filter_trend = trend_filter(data,trend_config)
        # filtered_trend_data = filter_trend.filter_trend_initial(
        #     initial_trend_data["trend_high"],
        #     initial_trend_data["trend_low"],
        # )
        # # 初始化趋势价格计算器
        # trend_tick_calculator = TrendTickCalculator(data, trend_config, filtered_trend_data)
        # trend_tick_data = trend_tick_calculator.update_trend_data(data, filtered_trend_data, data[-1, -1] + trend_config["interval"])
        # # trader = Trader(data, trend_config, trading_config)
        # # initial_filtered_trend_data = filtered_trend_data
        # # initial_filtered_trend_data["trend_high"] = filtered_trend_data["trend_high"][:-1]
        # # initial_filtered_trend_data["trend_low"] = filtered_trend_data["trend_low"][:-1]
        # plotter = Plotter(
        #         data[:-1],
        #         type_data[:-1],
        #         filtered_trend_data,
        #         len(data[:-1]),
        #         min(len(data) - 10, 500),
        #         10,
        #     )
        
        # plotter.enable_visualization = True
        # plotter.update_plot(
        #     filtered_trend_data,
        #     {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]},
        #     {"high_close":[], "low_close":[]},
        #     data[-1, -1],
        #     np.array([]),
        #     trend_tick_data,
        #     data,
        #     type_data,
        # )
        # plotter.run()
        # # while True:
        # #     plotter.run()
        
        # new_price = NewPrice(client, coin_type, contract_type)
        # order_manager = OrderManager()
        # trader = RealtimeTrader(data, trend_config, trading_config, order_manager)
        # trader.update_trend_price(data, trend_tick_data)
        
        # while True:
        #     try:
        #         current_tick_info = new_price.__next__()
        #     except Exception as e:
        #         print("获取实时价格失败，错误原因：",e)
        #         time.sleep(1)
        #         continue
        #     current_time = float(current_tick_info["time"])# 获取当前时间
        #     current_price = float(current_tick_info["price"])
            
        #     trader.judge_kline_signal(len(data), current_price)
        #     trader.open_close_signal(current_time, current_price)

        #     open_signal = trader.open_order_book
        #     order_response = open_order(open_signal, client, coin_type, contract_type, coin_info)
        #     if order_response is not None:
        #         print(order_response)
        #         trader.open_order_book = {"high_open":False, "low_open":False}
                
            
        #     if current_time - data[-1, 7] > time_number(interval) * 1000:
        #         # 新增K线数
        #         kline_num = int((current_time - data[-1, 7]) // (time_number(interval) * 1000) + 1)
        #         try:
        #             new_kline, new_type_data = get_latest_klines(client, coin_type, interval, 2)
        #         except Exception as e:
        #             print(e)
        #             time.sleep(1)
        #             continue
        #         if new_kline[0, 0] == data[-1, 0]: # 如果新k线与当前k线相同，则不更新数据
        #             continue
        #         data = np.concatenate((data, new_kline), axis=0)
        #         type_data = np.concatenate((type_data, new_type_data), axis=0)
        #         # 更新趋势数据
        #         trend_generator = backtester.run_backtest(data, type_data)
        #         current_trend = next(trend_generator)
        #         # 过滤趋势
        #         filtered_trend_data = filter_trend.process_new_trend(
        #             data,
        #             filtered_trend_data,
        #             current_trend,
        #         )
        #         # 计算趋势价格
        #         trend_tick_data = trend_tick_calculator.update_trend_data(data, filtered_trend_data, current_time)
        #         trader.update_trend_price(data, trend_tick_data)
                
                
        #         plotter.enable_visualization = True
        #         plotter.update_plot(
        #             filtered_trend_data,
        #             trader.open_signals,
        #             trader.close_signals,
        #             data[-1, -1],
        #             np.array([]),
        #             trend_tick_data,
        #             data,
        #             type_data,
        #         )

        #         plotter.run()
                
        #         if plotter.enable_visualization:
        #                 plotter.save_frame(coin_type)
                
                
        

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
        
        backtest_tick_path = f"backtest/{contract_type}/tick/{coin_type}"
        download_tick_data(backtest_tick_path, coin_type, contract_type, backtest_calculate_time_str, backtest_end_time_str)
        
        

        # 获取历史tick数据（回测时段的数据）
        backtest_tick_price = BacktestTickPriceManager(
            coin_type, backtest_calculate_time_str, backtest_end_time_str, contract_type
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
            client, coin_type, interval, length, backtest_start_time, backtest_end_time, contract_type
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

        parquet_filename = f"backtest/{contract_type}/tick/{coin_type}/parquet/{interval}"
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
            contract_type,
        )
        parquet_index = 1

        # TODO:在这里开始正式回测
        # 趋势参数
        if basic_config["exhaustive_mode"]:
            exhaustive_mode(basic_config, backtest_tick_price, data, type_data, base_trend_number, parquet_filename, parquet_index)
            return
        # else:
        #     return


        
        # 开始回测
        
        # 初始化回测器
        backtester = Backtester(data, type_data, base_trend_number)
        initial_trend_data = backtester.initial_trend_data
        # 过滤趋势
        filter_trend = trend_filter(data,trend_config)
        # 初始过滤趋势数据
        filtered_trend_data = filter_trend.filter_trend_initial(
            initial_trend_data["trend_high"],
            initial_trend_data["trend_low"],
        )  
        
        # 初始化趋势价格计算器
        trend_tick_calculator = TrendTickCalculator(data, trend_config, filtered_trend_data)
        
        order_manager = OrderManager()
        # # 初始化交易器
        trader = BacktestTrader(data, trend_config, trading_config, order_manager)

        # 6. 根据配置决定是否可视化
        if visualize_mode:
            cache_len = 1000  # 缓存长度
            if base_trend_number < visual_number:
                error_msg = f"base_trend_number 小于 visual_number，请检查配置文件"
                raise ValueError(error_msg)
            plotter = Plotter(
                data,
                type_data,
                filtered_trend_data,
                base_trend_number,
                visual_number,
                cache_len,
            )
            
            base_trend_number += 1
            # start_time = time.time()
            # backtester.run_backtest()
            # backtester.run_backtest()
            
            idx_draw = 0
            
            num_allow_draw = 0
            # HACK：这里开始循环回测
            with tqdm(desc="Backtesting Progress", mininterval=1) as pbar:
                for current_trend in backtester.run_backtest():
                    while plotter.paused:
                        plotter.run()
                        time.sleep(0.05)
                    
                    # 过滤趋势
                    filtered_trend_data = filter_trend.process_new_trend(
                        data,
                        filtered_trend_data,
                        current_trend,
                    )
                    
                    # 计算趋势价格
                    trend_tick_data = trend_tick_calculator.update_trend_data(data, filtered_trend_data, data[base_trend_number, 0], data[base_trend_number, 2])
                    
                    # 获取价格时间数组
                    current_timestamp = data[base_trend_number, 0]
                    price_time_array = backtest_tick_price.package_data_loader(parquet_filename, current_timestamp)
                    
                    trader.update_trend_price(data, trend_tick_data)
                    trader.judge_kline_signal(base_trend_number, data[base_trend_number, 1], data[base_trend_number, 3])
                    
                    trader.open_close_signal(price_time_array)
                    
                    if trader.paused:
                        plotter.enable_visualization = True
                    else:
                        plotter.enable_visualization = False
                        num_allow_draw = 0
                        
                    
                    # if trader.paused:
                    #     plotter.enable_visualization = True
                    #     num_allow_draw = 30  # 重置为30，确保后续30次迭代也会可视化
                    # elif num_allow_draw > 0:
                    #     plotter.enable_visualization = True
                    #     num_allow_draw -= 1  # 每次迭代减少计数
                    # else:
                    #     plotter.enable_visualization = False
                        
                    open_signals = trader.open_signals
                    close_signals = trader.close_signals
                    
                    # plotter.enable_visualization = True # 常开可视化
                    
                    # price_time_array = np.array([])
                    
                    plotter.update_plot(
                        current_trend,
                        open_signals,
                        close_signals,
                        data[base_trend_number, -1],
                        price_time_array,
                        trend_tick_data,
                        data,
                        type_data,
                    )
                    
                    plotter.run()
                    
                    if plotter.enable_visualization:
                        plotter.save_frame(coin_type)

                    base_trend_number += 1
                    
                    if base_trend_number >= data.shape[0]:
                        break


                    pbar.update(1)

            print("回测可视化已结束，继续后续操作...")
            
            # print(trader.order_book)
            visulize_orderbook(trader)
            # 计算盈利大于0.005的比例
            profits = [order['profit'] for order in trader.order_book]
            profitable_trades = sum(1 for profit in profits if profit > 0.005)
            total_trades = len(profits)
            profit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
            print(f"盈利大于0.005的比例: {profit_ratio:.2%}")
            
        else:
            # return
            base_trend_number += 1
            # start_time = time.time()
            with tqdm(desc="Backtesting Progress", mininterval=1) as pbar:
                for current_trend in backtester.run_backtest():
                    
                    # 过滤趋势
                    filtered_trend_data = filter_trend.process_new_trend(
                        data,
                        filtered_trend_data,
                        current_trend,
                    )
                    
                    # 计算趋势价格
                    trend_tick_data = trend_tick_calculator.update_trend_data(data, filtered_trend_data, data[base_trend_number, 0], data[base_trend_number, 2])
                    
                    # 获取价格时间数组
                    price_time_array = backtest_tick_price.package_data_loader(
                        parquet_index, parquet_filename
                    )
                    parquet_index += 1
                    
                    trader.open_close_signal(data, trend_tick_data, price_time_array, base_trend_number)
                    

                    open_signals = trader.open_signals
                    close_signals = trader.close_signals

                    base_trend_number += 1
                    if base_trend_number >= data.shape[0]:
                        break

                    pbar.update(1)
            profits = [order['profit'] for order in trader.order_book]
            profitable_trades = sum(1 for profit in profits if profit > 0.01)
            total_trades = len(profits)
            profit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
            print(f"盈利大于0.01的比例: {profit_ratio:.2%}")

def exhaustive_mode(basic_config, backtest_tick_price, data, type_data, base_trend_number, parquet_filename, parquet_index):
    import itertools
    import gc  # 导入垃圾回收模块
    delay_range = [10, 30, 50]
    filter_reverse_range = [True, False]
    min_line_age_range = [50, 100, 150]
    distance_threshold_range = [100, 200, 300]
    filter_trending_line_number_range = [5, 10]
    enter_threshold_range = [0.0003, 0.0006, 0.0009]
    leave_threshold_range = [0.0003, 0.0006, 0.0009]
    potential_profit_range = [0.002]
    trailing_profit_threshold_range = [0.002, 0.004, 0.006]
    trailing_stop_loss_range = [0.003, 0.005]
    best_profit = -np.inf
    best_params = None
    results = []
    # 保存初始的 base_trend_number 和 parquet_index，以便后续每轮都进行重置
    original_base_trend_number = base_trend_number
    original_parquet_index = parquet_index
    for delay, filter_reverse, min_line_age, distance_threshold, filter_trending_line_number, enter_threshold, leave_threshold, potential_profit, trailing_profit_threshold, trailing_stop_loss in tqdm(
        itertools.product(
            delay_range,
            filter_reverse_range,
            min_line_age_range,
            distance_threshold_range,
            filter_trending_line_number_range,
            enter_threshold_range,
            leave_threshold_range,
            potential_profit_range,
            trailing_profit_threshold_range,
            trailing_stop_loss_range
        ),
        desc="穷举中",
        total=len(delay_range) * len(filter_reverse_range) * len(min_line_age_range) * len(distance_threshold_range) * len(filter_trending_line_number_range) * len(enter_threshold_range) * len(leave_threshold_range) * len(potential_profit_range) * len(trailing_profit_threshold_range) * len(trailing_stop_loss_range)
    ):
        # 每次试验前重置 base_trend_number 和 parquet_index
        temp_base_trend_number = original_base_trend_number
        temp_parquet_index = original_parquet_index
        config = copy.deepcopy(basic_config)
        config["trend_config"]["delay"] = delay
        config["trend_config"]["filter_reverse"] = filter_reverse
        config["trend_config"]["min_line_age"] = min_line_age
        config["trend_config"]["distance_threshold"] = distance_threshold
        config["trend_config"]["filter_trending_line_number"] = filter_trending_line_number
        config["trading_config"]["enter_threshold"] = enter_threshold
        config["trading_config"]["leave_threshold"] = leave_threshold
        config["trading_config"]["potential_profit"] = potential_profit
        config["trading_config"]["trailing_profit_threshold"] = trailing_profit_threshold
        config["trading_config"]["trailing_stop_loss"] = trailing_stop_loss
        trend_config = config["trend_config"]
        trading_config = config["trading_config"]
        # 将趋势间隔时间转换为毫秒单位
        trend_config["interval"] = time_number(trend_config["interval"]) * 1000
        # 开始回测：使用临时的 temp_base_trend_number 而非外部的 base_trend_number
        backtester = Backtester(data, type_data, temp_base_trend_number)
        initial_trend_data = backtester.initial_trend_data
        # 过滤趋势
        filter_trend = trend_filter(data, trend_config)
        # 初始过滤趋势数据
        initial_filtered_trend_data = filter_trend.filter_trend_initial(
            initial_trend_data["trend_high"],
            initial_trend_data["trend_low"],
        )
        # 初始化趋势价格计算器
        trend_tick_calculator = TrendTickCalculator(data, trend_config, initial_filtered_trend_data)
        # 初始化交易器
        trader = BacktestTrader(data, trend_config, trading_config)
        # 运行回测，每次均使用 temp_base_trend_number 和 temp_parquet_index
        for current_trend in backtester.run_backtest():
            # 过滤趋势
            filtered_trend_data = filter_trend.process_new_trend(
                data,
                initial_filtered_trend_data,
                current_trend,
            )
            # 计算趋势价格
            trend_tick_data = trend_tick_calculator.update_trend_data(data, filtered_trend_data, data[temp_base_trend_number, 0], data[temp_base_trend_number, 2])
            # 获取价格时间数组
            price_time_array = backtest_tick_price.package_data_loader(
                temp_parquet_index, parquet_filename
            )
            temp_parquet_index += 1
            trader.open_close_signal(data[temp_base_trend_number, [1, 3, 4, 5]], trend_tick_data, price_time_array, temp_base_trend_number)
            temp_base_trend_number += 1
        profits = [order['profit'] for order in trader.order_book]
        profitable_trades = sum(1 for profit in profits if profit > 0.01)
        total_trades = len(profits)
        profit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
        print(f"盈利大于0.01的比例: {profit_ratio:.2%}")
        results.append({
            "trend_config": trend_config,
            "trading_config": trading_config,
            "profit_ratio": profit_ratio,
        })
        if profit_ratio > best_profit:
            best_profit = profit_ratio
            best_params = {
                "trend_config": trend_config,
                "trading_config": trading_config,
            }
            save_best_params(best_params)
        print(f"当前盈利比例: {profit_ratio:.2%}")
        # 显式删除不再使用的对象，并触发垃圾回收
        del backtester, trader, trend_tick_calculator, filter_trend, initial_filtered_trend_data
        gc.collect()
    print(f"最佳参数: {best_params}")
    print(f"最佳盈利比例: {best_profit:.2%}")
    
def save_best_params(best_params):
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

def visulize_orderbook(trader):
    order_book = trader.order_book
    import matplotlib.pyplot as plt
    from datetime import datetime  # 导入datetime模块

    # 提取订单信息
    order_ids = [order['order_id'] for order in order_book]
    stop_losses = [order['stop_loss'] for order in order_book]
    profits = [order['profit'] for order in order_book]
    tick_times = [order['tick_time'] for order in order_book]  # 提取tick_time

    # 将tick_time转换为可读格式，添加有效性检查
    readable_times = []
    for t in tick_times:
        if t >= 0:  # 检查时间戳是否有效
            readable_times.append(datetime.fromtimestamp(t / 1000).strftime('%d'))  # 转换为秒
        else:
            readable_times.append("Invalid Time")  # 无效时间处理

    # 计算累计利润
    cumulative_profits = [sum(profits[:i+1]) for i in range(len(profits))]

    

    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 绘制止损和盈利的订单
    for i in range(len(order_ids)):
        axs[0].scatter(readable_times[i], profits[i], color='green', label='Profit' if i == 0 else "")

    # 添加图例
    axs[0].legend()
    axs[0].set_title('Order Profit/Loss Visualization')
    axs[0].set_xlabel('Tick Time')  # 更新横坐标标签
    axs[0].set_ylabel('Profit')
    axs[0].axhline(0, color='black', linewidth=0.8, linestyle='--')  # 添加水平线表示盈亏平衡点
    axs[0].grid()

    # 绘制累计利润
    axs[1].plot(readable_times, cumulative_profits, color='blue', label='Cumulative Profit')  # 更新横坐标
    axs[1].set_title('Cumulative Profit Visualization')
    axs[1].set_xlabel('Tick Time')  # 更新横坐标标签
    axs[1].set_ylabel('Cumulative Profit')
    axs[1].axhline(0, color='black', linewidth=0.8, linestyle='--')  # 添加水平线表示盈亏平衡点
    axs[1].grid()

    # 显示图形
    plt.savefig("frames/orderbook.png")
    plt.show()
    
    
    # print(order_book)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", sort="cumulative")
    main()
    
# %%
