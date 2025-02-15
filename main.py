# %%
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


# %%


def main():
    # 记录获取历史数据的开始时间
    start_time = time.perf_counter()

    # 从配置文件中加载基本参数
    basic_config = load_basic_config("config/basic_config.yaml")

    key = basic_config["key"]
    secret = basic_config["secret"]
    coin_type = basic_config["coin_type"]
    aim_time_str = basic_config["aim_time"]
    total_length = basic_config["total_length"]
    interval = basic_config["interval"]

    run_type = basic_config["run_type"]

    trend_config = basic_config["trend_config"]
    trading_config = basic_config["trading_config"]
    trend_config["interval"] = time_number(trend_config["interval"]) * 1000

    # 创建 Binance 客户端

    client = Client(key, secret)
    if run_type:
        # 实盘交易模式
        current_price = get_current_price(client, coin_type)
        print(current_price)
        data, type_data = get_latest_klines(client, coin_type, interval, total_length)
        print(data[-1])

        # 初始化趋势数据
        trend_generator = backtest_calculate_trend_generator(
            data=data,
            initial_single_slope=initial_single_slope,
            calculate_trend=calculate_trend,
        )

    else:
        # 回测模式
        # 加载起始时间、结束时间、计算时间
        # backtest_start_time = basic_config["backtest_config"]["backtest_start_time"]
        # backtest_end_time = basic_config["backtest_config"]["backtest_end_time"]
        # backtest_calculate_time = basic_config["backtest_config"]["backtest_calculate_time"]
        
        # backtest_start_time = datetime.datetime.strptime(backtest_start_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
        # backtest_end_time = datetime.datetime.strptime(backtest_end_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
        # backtest_calculate_time = datetime.datetime.strptime(backtest_calculate_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
        
        # data, type_data = load_or_fetch_data(client, coin_type, interval, backtest_start_time, backtest_end_time, backtest_calculate_time)
        
        data, type_data = load_or_fetch_data(client, coin_type, interval, total_length)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"获取数据耗时: {elapsed_time:.6f} 秒")

        # 将目标时间字符串转换为毫秒时间戳
        # aim_time_str = basic_config["backtest_config"]["backtest_aim_time"]
        aim_time = (
            datetime.datetime.strptime(aim_time_str, "%Y-%m-%d %H:%M:%S").timestamp()
            * 1000
        )
        index = np.searchsorted(data[:, 0], aim_time, side="left")

        if index < len(data):
            print(
                f"First index with data[:,0] > aim_time is {index}, with timestamp {data[index, 0]}"
            )
        else:
            print("No data point found with data[:,0] > aim_time")

        visualize_mode = basic_config["visualize_mode"]
        visual_number = basic_config["visual_number"]
        base_trend_number = index  # 或者使用固定值，比如： base_trend_number = 10

        # 初始化趋势数据生成器
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
        

        if visualize_mode:  # 可视化模式切换
            delay = trend_config.get("delay")
            cache_len = 1000  # 缓存长度
            plotter = Plotter(
                data, type_data, initial_trend_data, visual_number, delay, cache_len
            )
            for current_trend in backtester.run_backtest():
                while plotter.paused:
                    plotter.run()
                    time.sleep(0.05)

                plotter.update_plot(current_trend)
                plotter.run()
                # time.sleep(0.5)
            print(
                "Backtest has ended with visualization. Continuing with post-Plotter operations..."
            )
        else:
            start_time = time.perf_counter()
            # 无可视化模式下的处理逻辑，示例中简单打印或记录趋势数据
            a = 1
            for current_trend in backtester.run_backtest():
                # 在这里你可以根据需要对 current_trend 进行处理或保存，而非更新图形
                # print("Processing trend:", current_trend)
                a += 1
            print(a)
            print(
                "Backtest has ended without visualization. Continuing with post-Backtester operations..."
            )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"回测耗时: {elapsed_time:.6f} 秒")


if __name__ == "__main__":
    main()
