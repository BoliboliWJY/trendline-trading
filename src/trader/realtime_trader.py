import numpy as np
# from src.order_book.order_booker import OrderBook
class RealtimeTrader:
    def __init__(self,data:np.ndarray, trend_config:dict, trading_config:dict, order_manager):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.order_manager = order_manager
        
        self.lock = False # 是否锁定信号
        self.base_trend_number = -1 # 初始化值
        
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
    
        self.open_signals = {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]}
        self.close_signals = {"high_close":[], "low_close":[]}
        
        self.open_order_book = {"high_open":False, "low_open":False} # 是否存在开仓订单
        
        self.close_potential_signal = False # 潜在平仓信号，需要具体对tick分析
        
        self.order_book = []
        
        self.paused = False # 是否暂停可视化更新
        
    def update_trend_price(self, data, trend_price:np.ndarray):
        """
        更新趋势价格
        """
        self.data = data
        self.trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        self.trend_price_low = trend_price["trend_price_low"][:, 1]
        
        self.trend_high_idx = 0 # 趋势线索引
        self.trend_low_idx = 0 # 趋势线索引

    def judge_kline_signal(self, base_trend_number, max_price, min_price = None):
        # 实盘状态下max_price输入为当前tick数据，不输入min_price，需要覆盖min_price
        if min_price is None: 
            min_price = max_price
        
        if self.base_trend_number != base_trend_number: # 更新了data索引
            self.base_trend_number = base_trend_number
            self.lock = False
        
        if self.lock: # 信号被锁定，不进行信号判断
            return
        
        self.paused = False
        sell_enter_val = 1 - max_price / self.trend_price_high[0] if self.trend_price_high.size > 0 else self.trading_config["enter_threshold"] + 1
        sell_profit_val = 1 - self.trend_price_low[0] / max_price if self.trend_price_low.size > 0 else 1
        buy_enter_val = 1 - self.trend_price_low[0] / min_price if self.trend_price_low.size > 0 else self.trading_config["enter_threshold"] + 1
        buy_profit_val = 1 - min_price / self.trend_price_high[0] if self.trend_price_high.size > 0 else 1
        
        if sell_enter_val < self.trading_config["enter_threshold"] and sell_profit_val > self.trading_config["potential_profit"]:
            # self.paused = True
            self.sell_open_potential_signal = True
        elif buy_enter_val < self.trading_config["enter_threshold"] and buy_profit_val > self.trading_config["potential_profit"]:
            # self.paused = True
            self.buy_open_potential_signal = True
        else:
            self.paused = False
            self.buy_open_potential_signal = False
            self.sell_open_potential_signal = False
            
    def open_close_signal(self, current_time, current_price):
        time = current_time
        tick_price = current_price
        
        if self.sell_open_potential_signal or self.sell_open_potential_signal_last:
            self.lock = True # 锁定信号
            self.trend_high_idx, self.sell_open_potential_signal_last = self.process_open_signal(self.trend_price_high, tick_price, time, "SELL", "high_open", self.trend_high_idx)
            
        if self.buy_open_potential_signal or self.buy_open_potential_signal_last:
            self.lock = True # 锁定信号
            self.trend_low_idx, self.buy_open_potential_signal_last = self.process_open_signal(self.trend_price_low, tick_price, time, "BUY", "low_open", self.trend_low_idx)
            
        
    def process_open_signal(self, trend_price, tick_price, time, side, open_signal_key, trend_idx):
        """
        处理开仓信号

        Args:
            trend_price: 趋势价格数组
            tick_price: tick价格数组
            time: 时间数组
            side: "BUY" 或 "SELL"，对应订单类型
            open_signal_key: "high_open" 或 "low_open"
            trend_idx: 趋势线索引
        """
        last_signal = False # 初始化last_signal
        # True为发生，False为未发生
        break_signal, trend_idx = self.judge_signal(trend_price, tick_price, trend_idx, side, "break") # 打破趋势线信号
        leave_signal, trend_idx = self.judge_signal(trend_price, tick_price, trend_idx, side, "leave") # 离开开仓信号
        
        if break_signal: # 打破趋势线
            self.lock = False # 不锁定信号
            return trend_idx, last_signal # 返回新趋势线下标
        
        if leave_signal: # 出现开仓信号, 执行开仓操作
            current_time = time
            current_price = tick_price
            self.open_signals[open_signal_key].append((current_time, current_price)) # 记录开仓信号,用于绘图
            self.open_order_book[open_signal_key] = True # 记录开仓订单,用于实际开仓
            if side == "SELL":
                self.sell_open_potential_signal_last = False
                
            elif side == "BUY":
                self.buy_open_potential_signal_last = False
            
            self.lock = False # 关闭锁定信号
        else: # 没有出现开仓信号
            last_signal = True # 上一根k线有开仓信号
        
        return trend_idx, last_signal
        
    def judge_signal(self, trend_price, tick_price, trend_idx, side, mode="break"):
        """
        判断离开信号
        """
        if side == "SELL":
            ratio = (trend_price[trend_idx] - tick_price) / tick_price
        else:
            ratio = (tick_price - trend_price[trend_idx]) / tick_price
        
        if mode == "break": # 打破趋势线信号
            if ratio < 2e-4:
                # 出现打破趋势价格的情况，更新趋势线下标并重置进入标志
                trend_idx += 1
                return True, trend_idx
            else:
                return False, trend_idx
        elif mode == "leave":
            if ratio > self.trading_config["leave_threshold"]:
                return True, trend_idx
            else:
                return False, trend_idx
        
        
        
        
        
        
        
        
        
        
