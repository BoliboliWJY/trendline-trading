# %%
import time
import datetime
import numpy as np
from binance.spot import Spot as Client
# from binance import Client
from src.config_manager import load_basic_config
from src.data_loader import load_or_fetch_data
from src.trend_generator import backtest_calculate_trend_generator
from src.trend_process import initial_single_slope, calculate_trend
from src.filter.filters import filter_trend
from src.time_number import time_number
from src.plot_figure.plotter import Plotter
from src.latest_data.new_price import get_current_price
from src.latest_data.latest_klines import get_latest_klines


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
        # 获取或加载历史数据
        data, type_data = load_or_fetch_data(client, coin_type, interval, total_length)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"获取数据耗时: {elapsed_time:.6f} 秒")

        # 将目标时间字符串转换为毫秒时间戳
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

        visual_number = basic_config["visual_number"]
        base_trend_number = (
            index - visual_number
        )  # 或者使用固定值，比如： base_trend_number = 10

        # 初始化趋势数据生成器
        trend_generator = backtest_calculate_trend_generator(
            data=data,
            initial_single_slope=initial_single_slope,
            calculate_trend=calculate_trend,
        )

        # 初始化 Plotter 类用于回测数据可视化
        plotter_app = Plotter(
            data,
            type_data,
            trend_generator,
            filter_trend,
            trend_config,
            trading_config,
            basic_config,
            base_trend_number=base_trend_number,
            visual_number=visual_number,
            update_interval=20,
            cache_size=visual_number * 2,
        )

        # 运行 Plotter
        plotter_app.run()

        print("Plotter has ended. Continuing with post-Plotter operations...")


if __name__ == "__main__":
    main()
