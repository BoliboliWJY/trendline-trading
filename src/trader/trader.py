import numpy as np
from src.utils import profile_method

# @profile_method
class Trader:
    def __init__(self,data:np.ndarray, trend_config:dict, trading_config:dict):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        
        self.open_potential_signal = False # 潜在开仓信号，需要具体对tick分析
        self.buy_open_potential_signal = False # 潜在买入开仓信号
        self.sell_open_potential_signal = False # 潜在卖出开仓信号
        
        self.close_potential_signal = False # 潜在平仓信号，需要具体对tick分析
        
        self.paused = False # 是否暂停可视化更新
        
    def open_close_signal(self,max_min_price:np.ndarray, trend_price:np.ndarray, price_time_array:np.ndarray):
        """
        根据趋势价格和价格时间数组，生成开仓和平仓信号
        """
        self.max_price = max_min_price[0]
        self.min_price = max_min_price[1]
        # trend_price_high = trend_price["trend_price_high"]
        # trend_price_low = trend_price["trend_price_low"]
        # trend_high = trend_price["trend_high"] # 趋势线斜率，截距
        # trend_low = trend_price["trend_low"]
        trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        trend_price_low = trend_price["trend_price_low"][:, 1]
        price_time_array = price_time_array
        time = price_time_array[:, 0]
        tick_price = price_time_array[:, 1]
        
        # trend_price_high_tick = self.compute_trend_price(trend_high, time) # 对应tick下的价格
        # trend_price_low_tick = self.compute_trend_price(trend_low, time)
        
        # if not self.open_potential_signal: # 基于k线价格简单判断
        # 做空进入阈值
        sell_enter_val = 1 - self.max_price / trend_price_high[0] if trend_price_high.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做空潜在利润值
        sell_profit_val = 1 - trend_price_low[0] / self.max_price if trend_price_low.size > 0 else 1
        # 做多进入阈值
        buy_enter_val = 1 - trend_price_low[0] / self.min_price if trend_price_low.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做多潜在利润值
        buy_profit_val = 1 - self.min_price / trend_price_high[0] if trend_price_high.size > 0 else 1
        
        if sell_enter_val < self.trading_config["enter_threshold"] and sell_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在卖出开仓信号, 进入阈值/打破趋势线")
            # print("最高价格", self.max_price, "趋势价格", trend_price_high[0, 1], "相差", sell_enter_val)
            self.paused = True # 暂停可视化
            self.open_potential_signal = True # 出现潜在开仓信号
            self.sell_open_potential_signal = True # 出现潜在卖出开仓信号
            self.buy_open_potential_signal = False # 重置潜在买入开仓信号
        elif buy_enter_val < self.trading_config["enter_threshold"] and buy_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在买入开仓信号, 进入阈值/打破趋势线")
            # print("最低价格", self.min_price, "趋势价格", trend_price_low[0, 1], "相差", buy_enter_val)
            self.paused = True # 暂停可视化
            self.open_potential_signal = True # 出现潜在开仓信号
            self.buy_open_potential_signal = True # 出现潜在买入开仓信号
            self.sell_open_potential_signal = False # 重置潜在卖出开仓信号
        else: # 如果啥信号也没有就重置，避免影响非连续的点
            self.paused = False
            self.open_potential_signal = False
            self.buy_open_potential_signal = False
            self.sell_open_potential_signal = False
                
        # TODO: 注意现在仅用k线对应的趋势价格判断，没有精确到tick
        # if self.sell_open_potential_signal: # 如果出现潜在卖出开仓信号
        #     trend_idx = 0
        #     nearest_trend_price = trend_price_high[trend_idx]
        #     threshold = nearest_trend_price / (1 + self.trading_config["enter_threshold"])
        #     valid_indices = np.where(tick_price >= threshold)[0]
            
        # if self.buy_open_potential_signal: # 如果出现潜在买入开仓信号
        #     trend_idx = 0
        #     nearest_trend_price = trend_price_low[trend_idx]
        #     threshold = nearest_trend_price / (1 - self.trading_config["enter_threshold"])
        #     valid_indices = np.where(tick_price <= threshold)[0][0]
            
            
                
        # trend_price_high = self.compute_trend_price(trend_price_high, time)
        # trend_price_low = self.compute_trend_price(trend_price_low, time)
    
    def compute_trend_price(self,trend, tick_time):
        """计算趋势价格"""
        k = trend[:, 0][:, None]
        b = trend[:, 1][:, None]
        result = k * tick_time + b
        return result

    

    def judge_sell_open_tick(self, trend_price_high, price_time_array):
        """卖出开仓"""
        trend_idx = 0
        enter_signal = False # 是否进入范围
        leave_signal = False # 是否离开范围(确实反弹)
        open_sell_signal = False # 是否开仓
        for i in range(len(price_time_array)):
            if not enter_signal: # 如果还未进入范围
                if self.calculate_rate(price_time_array, trend_price_high, i, trend_idx) < self.trading_config["enter_threshold"]:
                    enter_signal = True
                    
            else: # 如果已经进入范围
                if self.calculate_rate(price_time_array, trend_price_high, i, trend_idx) < 0:# 打破了当前趋势线
                    enter_signal = False
                    trend_idx += 1
                elif self.calculate_rate(price_time_array, trend_price_high, i, trend_idx) > self.trading_config["leave_threshold"]:
                    leave_signal = True # 离开趋势线阈值范围内
                    open_sell_signal = True # 卖出开仓信号出现
                    
                    yield open_sell_signal
                
            
        pass
    
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

            