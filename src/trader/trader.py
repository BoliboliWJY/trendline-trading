import numpy as np
from src.utils import profile_method

# @profile_method
class Trader:
    def __init__(self,data:np.ndarray, trend_config:dict, trading_config:dict):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        
        self.buy_open_potential_signal = False # 潜在买入开仓信号
        self.sell_open_potential_signal = False # 潜在卖出开仓信号
        self.buy_open_potential_signal_last = False # 上一根k线的潜在买入开仓信号
        self.sell_open_potential_signal_last = False # 上一根k线的潜在卖出开仓信号
        
        self.reserved_buy_open_signal = False # 保留买入开仓信号
        self.reserved_sell_open_signal = False # 保留卖出开仓信号
        
        self.high_signal_found = False # 高趋势线信号找到
        self.low_signal_found = False # 低趋势线信号找到
        
        self.trend_price_high_last = 0 # 上一次趋势线最近的高点
        self.trend_price_low_last = 0 # 上一次趋势线最近的低点
    
        self.open_signals = {"high_open":[], "low_open":[]}
        self.close_signals = {"high_close":[], "low_close":[]}
        
        self.close_potential_signal = False # 潜在平仓信号，需要具体对tick分析
        
        self.order_book = []
        
        self.paused = False # 是否暂停可视化更新
        
    def open_close_signal(self,max_min_price:np.ndarray, trend_price:np.ndarray, price_time_array:np.ndarray):
        """
        根据趋势价格和价格时间数组，生成开仓和平仓信号
        """
        # trend_price_high = trend_price["trend_price_high"]
        # trend_price_low = trend_price["trend_price_low"]
        # trend_high = trend_price["trend_high"] # 趋势线斜率，截距
        # trend_low = trend_price["trend_low"]
        trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        trend_price_low = trend_price["trend_price_low"][:, 1]
        price_time_array = price_time_array
        time = price_time_array[:, 0]
        tick_price = price_time_array[:, 1]
        
        max_price = np.max(tick_price)
        max_index = np.argmax(tick_price)
        max_after_min = np.min(tick_price[max_index:])
        min_price = np.min(tick_price)
        min_index = np.argmin(tick_price)
        min_after_max = np.max(tick_price[min_index:])
        start_price = tick_price[0]
        end_price = tick_price[-1]
        
        self.paused = False # 重置暂停状态
        
        # trend_price_high_tick = self.compute_trend_price(trend_high, time) # 对应tick下的价格
        # trend_price_low_tick = self.compute_trend_price(trend_low, time)
        
        # if not self.open_potential_signal: # 基于k线价格简单判断
        # 做空进入阈值
        sell_enter_val = 1 - max_price / trend_price_high[0] if trend_price_high.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做空潜在利润值
        sell_profit_val = 1 - trend_price_low[0] / max_price if trend_price_low.size > 0 else 1
        # 做多进入阈值
        buy_enter_val = 1 - trend_price_low[0] / min_price if trend_price_low.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做多潜在利润值
        buy_profit_val = 1 - min_price / trend_price_high[0] if trend_price_high.size > 0 else 1
        
        if sell_enter_val < self.trading_config["enter_threshold"] and sell_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在卖出开仓信号, 进入阈值/打破趋势线")
            # print("最高价格", self.max_price, "趋势价格", trend_price_high[0, 1], "相差", sell_enter_val)
            # self.paused = True # 暂停可视化
            self.sell_open_potential_signal = True # 出现潜在卖出开仓信号
            self.buy_open_potential_signal = False # 重置潜在买入开仓信号
            # self.sell_open_potential_signal_last = True
            # self.buy_open_potential_signal_last = False
        elif buy_enter_val < self.trading_config["enter_threshold"] and buy_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在买入开仓信号, 进入阈值/打破趋势线")
            # print("最低价格", self.min_price, "趋势价格", trend_price_low[0, 1], "相差", buy_enter_val)
            # self.paused = True # 暂停可视化
            self.buy_open_potential_signal = True # 出现潜在买入开仓信号
            self.sell_open_potential_signal = False # 重置潜在卖出开仓信号
            # self.buy_open_potential_signal_last = True
            # self.sell_open_potential_signal_last = False
        else: # 如果啥信号也没有就重置，避免影响非连续的点
            self.paused = False
            self.buy_open_potential_signal = False
            self.sell_open_potential_signal = False
            # self.buy_open_potential_signal_last = False
            # self.sell_open_potential_signal_last = False
        

        if self.sell_open_potential_signal or self.sell_open_potential_signal_last: # 如果出现潜在卖出开仓信号
            self.process_sell_open_signal(trend_price_high, tick_price, self.sell_open_potential_signal_last, time)
            
        if self.buy_open_potential_signal or self.buy_open_potential_signal_last: # 如果出现潜在买入开仓信号
            self.process_buy_open_signal(trend_price_low, tick_price, self.buy_open_potential_signal_last, time)
                    
                    
    def process_sell_open_signal(self, trend_price_high, tick_price, last_signal, time):
        start_index = 0
        trend_idx = 0
        self.open_signals["high_open"] = []
        while start_index < len(tick_price):
             # 1. 等待进入趋势线区域（即价格接近趋势线，在预设enter_threshold之下）
            if not last_signal: # 如果上一次没有开仓,但是有进入过趋势线
                enter_signal, start_index, trend_idx = self.judge_signal(trend_price_high, tick_price, start_index, trend_idx, self.high_signal_found, direction="high", mode="enter")
                if start_index is None or trend_idx is None or not enter_signal:
                    break
            
            # 2. 等待离开趋势线区域（即价格远离趋势线，在预设leave_threshold之上）
            leave_signal, start_index, trend_idx = self.judge_signal(trend_price_high, tick_price, start_index, trend_idx, self.high_signal_found, direction="high", mode="leave")
            if not leave_signal and start_index is not None:
                continue
            if start_index is None or trend_idx is None:
                last_signal = True # 没有开仓，但是已经进入过趋势线
                break
            if leave_signal:
                # 出发开仓订单
                self.paused = True
                self.sell_open_time = time[start_index] # 获取开仓时间
                self.sell_open_price = tick_price[start_index] # 获取开仓价格
                
                self.place_order("sell", self.sell_open_price, self.sell_open_time)
                self.open_signals["high_open"].append([self.sell_open_time, self.sell_open_price])

                self.sell_open_potential_signal_last = False # 重置
                
                last_signal = False
                continue
            else:
                pass # 暂时没别的操作
        self.sell_open_potential_signal_last = last_signal
        
    def process_buy_open_signal(self, trend_price_low, tick_price, last_signal, time):
        """买入开仓
            Args:
                trend_price_low (_type_): 趋势线低点
                tick_price (_type_): 价格
                last_signal (_type_): 上次有没有进入阈值范围内
                time (_type_): 时间
        """
        start_index = 0
        trend_idx = 0
        self.open_signals["low_open"] = []
        while start_index < len(tick_price):
            if not last_signal:
                enter_signal, start_index, trend_idx = self.judge_signal(trend_price_low, tick_price, start_index, trend_idx, self.low_signal_found, direction="low", mode="enter")
                if start_index is None or trend_idx is None or not enter_signal:
                    last_signal = False # 如果未进入阈值范围内，则重置
                    break
                
            leave_signal, start_index, trend_idx = self.judge_signal(trend_price_low, tick_price, start_index, trend_idx, self.low_signal_found, direction="low", mode="leave")
            if not leave_signal and start_index is not None:# 打破阈值但未循环结束
                last_signal = False
                continue
            if start_index is None or trend_idx is None:
                last_signal = True # 没有开仓，但是已经进入过阈值范围内
                break
            if leave_signal:
                self.paused = True
                self.buy_open_time = time[start_index]
                self.buy_open_price = tick_price[start_index]
                
                self.place_order("buy", self.buy_open_price, self.buy_open_time)
                self.open_signals["low_open"].append([self.buy_open_time, self.buy_open_price])
                
                self.buy_open_potential_signal_last = False # 完成本次开仓判断，完成开仓
                
                last_signal = False
                continue
            else:
                pass # 暂时没别的操作
        self.buy_open_potential_signal_last = last_signal
                
        
        
    def place_order(self, order_type:str, price:float, tick_time):
        order_id = len(self.order_book) + 1
        order = {
            "order_id": order_id,
            "order_type": order_type,
            "price": price,
            "tick_time": tick_time,
            "status": "open"
        }
        self.order_book.append(order)
        print(f"开仓订单: {order}")

    
        
    def judge_signal(self, trend_prices, tick_prices, start_index, trend_idx, signal_found, direction:str, mode = "enter"):
        """
        根据模式判断是否进入或离开信号区域
        mode: "enter" 表示判断进入候选区域；"leave" 表示判断离开后触发信号
        """
        # signal_found = False
        while start_index < len(tick_prices):
            if trend_idx >= len(trend_prices):
                break
            current_price = tick_prices[start_index]
            # ratio = 1 - current_price / trend_prices[trend_idx]
            if direction == "high":
                ratio = (trend_prices[trend_idx] - current_price) / current_price
            else:
                ratio = (current_price - trend_prices[trend_idx]) / current_price
            if mode == "enter":
                if not signal_found and ratio < self.trading_config["enter_threshold"]:
                    signal_found = True
                    return True, start_index, trend_idx
                elif signal_found:
                    if ratio < 0:
                        # 出现打破趋势价格的情况，更新趋势线下标并重置进入标志
                        trend_idx += 1
                        signal_found = False
                    

            elif mode == "leave":
                if ratio > self.trading_config["leave_threshold"]:
                    return True, start_index, trend_idx
                elif ratio < 0:
                    # 出现打破趋势价格的情况，更新趋势线下标并重置进入标志
                    trend_idx += 1
                    signal_found = False
                    return False, start_index, trend_idx # 打破阈值但未循环结束
            start_index += 1
        return False, None, None # 啥也没找到
        
        
        
        
        
    
    def judge_buy_open_tick(self, trend_price_low, price_time_array):
        """买入开仓"""
        trend_idx = 0
        enter_signal = False # 是否进入范围
        leave_signal = False # 是否离开范围(确实反弹)
        open_buy_signal = False # 是否开仓
        for i in range(len(price_time_array)):
            if not enter_signal: # 如果还未进入范围
                if self.calculate_rate(trend_price_low, price_time_array, trend_idx, i) < self.trading_config["enter_threshold"]:
                    enter_signal = True
                    rate = self.calculate_rate(trend_price_low, price_time_array, trend_idx, i)
            else: # 如果已经进入范围
                if self.calculate_rate(trend_price_low, price_time_array, trend_idx, i) < 0:
                    enter_signal = False
                    trend_idx += 1
                elif self.calculate_rate(trend_price_low, price_time_array, trend_idx, i) > self.trading_config["leave_threshold"]:
                    leave_signal = True # 离开趋势线阈值范围内
                    open_buy_signal = True # 买入开仓信号出现
                    print("买入开仓, 相差", rate)
                    self.plotter.paused = True
                    
                    yield open_buy_signal
        
        
        pass
    
    
    def calculate_rate(self, low_obj, high_obj, low_idx, high_idx):
        """计算低点与低趋势之间的价格比值"""
        return 1 - low_obj[low_idx, 1] / high_obj[high_idx, 1]



    def compute_trend_price(self,trend, tick_time):
        """计算趋势价格"""
        k = trend[:, 0][:, None]
        b = trend[:, 1][:, None]
        result = k * tick_time + b
        return result
    
    
            