import threading
from typing import Dict, List, Any
import time
import numpy as np

from src.backtester.backtester import Backtester
from src.filter.filters import trend_filter
from src.time_number import time_number
from src.latest_data.new_price import NewPrice
from src.latest_data.latest_klines import get_latest_klines
from src.plotter.plotter import Plotter
from src.coin_info.coin_info import CoinInfo
from src.trader.realtime_trader import RealtimeTrader
from src.order_book import open_order, OrderManager
from src.backtester.trend_tick_calculator import TrendTickCalculator


class CoinMonitor:
    def __init__(self, client, coin_type: str, contract_type: str, interval: str, trend_config, trading_config, coin_info):
        self.client = client
        self.coin_type = coin_type
        self.contract_type = contract_type
        self.interval = interval
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.coin_info = coin_info
        self.running = False
        self.plotter = None
        self.thread = None

    def initialize(self, total_length:int):
        # 初始化数据,获取最新K线数据
        self.data, self.type_data = get_latest_klines(self.client, self.coin_type, self.interval, total_length + 1)
        
        # 初始化趋势生成器
        self.backtester = Backtester(
            self.data, 
            self.type_data
        )
        initial_trend_data = self.backtester.initial_trend_data
        
        self.filter_trend = trend_filter(
            self.data, 
            self.trend_config
        )
        self.filtered_trend_data = self.filter_trend.filter_trend_initial(
            initial_trend_data["trend_high"],
            initial_trend_data["trend_low"],
        )

        # 初始化趋势价格计算器
        self.trend_tick_calculator = TrendTickCalculator(
            self.data, 
            self.trend_config, 
            self.filtered_trend_data
            )
        
        
        self.trend_tick_data = self.trend_tick_calculator.update_trend_data(
            self.data,
            self.filtered_trend_data,
            self.data[-1, -1] + self.trend_config["interval"]
        )
        
        # 初始化绘图器
        self.plotter = Plotter(
            self.data[:-1],
            self.type_data[:-1],
            self.filtered_trend_data,
            len(self.data[:-1]),
            min(len(self.data) - 10, 500),
            10,
        )
        
        # 初始化交易器
        self.new_price = NewPrice(
            self.client,
            self.coin_type,
            self.contract_type,
        )
        self.order_manager = OrderManager()
        self.trader = RealtimeTrader(
            self.data,
            self.trend_config,
            self.trading_config,
            self.order_manager,
        )
        self.trader.update_trend_price(
            self.data,
            self.trend_tick_data,
        )
        
        self.plotter.enable_visualization = True
        self.plotter.update_plot(
            self.filtered_trend_data,
            {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]},
            {"high_close":[], "low_close":[]},
            self.data[-1, -1],
            np.array([]),
            self.trend_tick_data,
            self.data,
            self.type_data,
        )
        
        self.run_visulization()
        
        
    def run_visulization(self):
        # 运行可视化
        self.plotter.run()
        self.plotter.save_frame(self.coin_type)
                
    def moniter_loop(self):
        # 监控循环
        self.running = True
        while self.running:
            try:
                # 获取最新价格
                current_tick_info = self.new_price.__next__()
                current_time = float(current_tick_info["time"])
                current_price = float(current_tick_info["price"])
                
                # 判断信号并处理
                self.trader.judge_kline_signal(len(self.data), current_price)
                self.trader.open_close_signal(current_time, current_price)
                
                # 处理开仓信号
                open_signal = self.trader.open_order_book
                order_response = open_order(open_signal, self.client, self.coin_type, self.contract_type, self.coin_info)
                if order_response is not None:
                    print(order_response)
                    self.trader.open_order_book = {"high_open":False, "low_open":False}
                
                print(f"[{self.coin_type}] 当前时间: {current_time}")
                print(current_time - self.data[-1, 7])
                
                # 判断是否需要更新趋势
                if current_time - self.data[-1, 7] > time_number(self.interval) * 1000:
                    self.update_klines(current_time)
            
            except Exception as e:
                print(f"[{self.coin_type}] 监控循环发生错误: {e}")
                time.sleep(1)
                
    def update_klines(self, current_time):
        
        # 更新K线数据
        try:
            # BUG: 这里有问题，虽然确实只需要一个最新数据线，但很有可能出现网络问题时，只获取最新数据，缺失的k线被遗忘了
            kline_num = int((current_time - self.data[-1, 7]) // self.trend_config["interval"]) + 1
            print(kline_num)
            new_kline, new_type_data = get_latest_klines(self.client, self.coin_type, self.interval, kline_num)
            
            # 如果新K线与当前K线相同，则不更新数据
            if new_kline[0, 0] == self.data[-1, 0]:
                return
            print(f"[{self.coin_type}] 更新K线数据")
            print(self.trend_config["interval"])
            print(new_kline[0, -1] + self.trend_config["interval"] - time.time() * 1000)
            # 更新数据
            self.data = np.concatenate((self.data, new_kline), axis=0)
            self.type_data = np.concatenate((self.type_data, new_type_data), axis=0)
            
            # 更新趋势数据
            trend_generator = self.backtester.run_backtest(self.data, self.type_data)
            for i in range(kline_num - 1):  
                current_trend = next(trend_generator)
                # 过滤趋势
                self.filtered_trend_data = self.filter_trend.process_new_trend(
                    self.data,
                    self.filtered_trend_data,
                    current_trend,
                )
            # 计算趋势价格
            self.trend_tick_data = self.trend_tick_calculator.update_trend_data(
                self.data,
                self.filtered_trend_data,
                current_time,
            )
            self.trader.update_trend_price(
                self.data,
                self.trend_tick_data,
            )
            # 更新绘图器
            self.plotter.enable_visualization = True
            self.plotter.update_plot(
                self.filtered_trend_data,
                self.trader.open_signals,
                self.trader.close_signals,
                self.data[-1, -1],
                np.array([]),
                self.trend_tick_data,
                self.data,
                self.type_data,
            )
            print(f"[{self.coin_type}] 更新绘图器")

            self.run_visulization()
        
        except Exception as e:
            print(f"[{self.coin_type}] 更新K线数据发生错误: {e}")
            
    def start(self):
        # 启动监控
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.moniter_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"[{self.coin_type}] 监控已启动")
            return True
        return False

    def stop(self):
        # 停止监控
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout = 2.0)
            print(f"[{self.coin_type}] 监控已停止")
            
class CoinMonitorManager:
    # 管理多个CoinMonitor实例
    def __init__(self, client, trend_config, trading_config):
        self.client = client
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.moniters: Dict[str, CoinMonitor] = {}
        
    def add_coin(self, coin_type: str, contract_type, interval, total_length):
        # 添加新的监控实例
        if coin_type in self.moniters:
            print(f"[{coin_type}] 已存在监控实例")
            return
        coin_info = CoinInfo(self.client, coin_type, self.trading_config["min_amount"])
        monitor = CoinMonitor(
            self.client,
            coin_type,
            contract_type,
            interval,
            self.trend_config,
            self.trading_config,
            coin_info,
        )
        monitor.initialize(total_length)
        
        # 保存监控器
        self.moniters[coin_type] = monitor
        print(f"[{coin_type}] 监控实例已添加")
        
    def start_all(self):
        # 启动所有监控实例
        for moniter in self.moniters.values():
            moniter.start()
            
    def stop_all(self):
        # 停止所有监控实例
        for moniter in self.moniters.values():
            moniter.stop()
            
    def start_coin(self, coin_type: str):
        # 启动指定币种的监控实例
        if coin_type not in self.moniters:
            print(f"[{coin_type}] 监控实例不存在")
            return 
        self.moniters[coin_type].start()
        
    def stop_coin(self, coin_type: str):
        # 停止指定币种的监控实例
        if coin_type not in self.moniters:
            print(f"[{coin_type}] 监控实例不存在")
            return
        self.moniters[coin_type].stop()
        
    def get_status(self):
        # 获取所有监控实例的状态
        return {coin_type: moniter.running for coin_type, moniter in self.moniters.items()}
    
    def run_visualization(self, coin_type = None):
        # 运行指定币种的绘图
        if coin_type is None:
            for moniter in self.moniters.values():
                moniter.run_visulization()
        else:
            for moniter in self.moniters.values():
                moniter.run_visulization()
            
        
            
                    
        
        
        
        
        
        
