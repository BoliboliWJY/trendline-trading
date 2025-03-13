import numpy as np
from src.order_book.order_booker import OrderBook
class RealtimeTrader:
    def __init__(self, data, trend_config, trading_config, order_manager):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.order_manager = order_manager
        
        # 初始化趋势价格数组
        self.trend_price_high = np.array([])  # 高趋势价格数组
        self.trend_price_low = np.array([])   # 低趋势价格数组
        
        # 用于记录候选信号（单tick候选）——实时场景下需要跨tick维护状态
        self.sell_candidate = None  
        self.buy_candidate = None
        
        # 用于记录开仓和平仓信号,方便可视化
        self.open_signals = {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]}
        self.close_signals = {"high_close":[], "low_close":[]}
        
    def update_trend_price(self, data, trend_price:np.ndarray):
        """
        更新趋势价格
        """
        self.data = data
        self.trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        self.trend_price_low = trend_price["trend_price_low"][:, 1]
        
    def open_close_signal(self, current_tick_time, current_tick_price):
        """
        根据趋势价格和价格时间数组，生成开仓和平仓信号
        此处不更新趋势价格，仅根据当前价格生成信号
        """
        time = current_tick_time
        tick_price = current_tick_price
        
        # 做空进入阈值  
        sell_enter_val = 1 - tick_price / self.trend_price_high[0] if self.trend_price_high.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做空潜在利润值
        sell_profit_val = 1 - self.trend_price_low[0] / tick_price if self.trend_price_low.size > 0 else 1
        # 做多进入阈值
        buy_enter_val = 1 - self.trend_price_low[0] / tick_price if self.trend_price_low.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做多潜在利润值
        buy_profit_val = 1 - tick_price / self.trend_price_high[0] if self.trend_price_high.size > 0 else 1
        
        if sell_enter_val < self.trading_config["enter_threshold"] and sell_profit_val > self.trading_config["potential_profit"]:
            if self.sell_candidate is None:
                self.sell_candidate = {"time":time, "price":tick_price}
            else:
                # 检查是否满足离场条件（比如突破离场阈值），也可结合其他条件确认
                # 此处示例：当当前 tick 超过离场阈值时触发下单
                if (tick_price - self.trend_price_high[0]) / tick_price > self.trading_config["leave_threshold"]:
                    # 模拟下单
                    order_id = (
                        len(self.order_manager.pending_orders) +
                        len(self.order_manager.completed_orders) + 1
                    )
                    stop_loss = tick_price * (1 + self.trading_config["stop_loss"])
                    take_profit = tick_price * (1 - self.trading_config["take_profit"])
                    order = OrderBook(
                        order_id=order_id,
                        order_type="sell",
                        open_price=self.sell_candidate["price"],
                        quantity=100,
                        open_time=self.sell_candidate["time"],
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    self.order_manager.add_order(order)
                    self.sell_candidate = None # 清空候选信号
        else:
            self.sell_candidate = None
                
        
            
    def process_open_signal(self, trend_prices, tick_prices, last_signal, time, order_type:str, open_signal_key, signal_found):
        """
        处理开仓信号
        """
        trend_idx = 0
        if not last_signal:
            enter_signal, start_index, trend_idx = self.judge_signal(trend_prices, tick_prices, trend_idx, signal_found, order_type, mode="enter")
            if start_index is None or trend_idx is None:
                last_signal = False # 没有进入趋势线，且循环结束
                return
            if enter_signal: # 进入阈值
                self.paused = True # 进入阈值，绘图
                current_time = time
                current_price = tick_prices[start_index]
                self.open_signals[open_signal_key + "_enter"].append([current_time, current_price])
            
        leave_signal, trend_idx = self.judge_signal(trend_prices, tick_prices, trend_idx, signal_found, order_type, mode="leave")
        if not leave_signal is not None: # 打破阈值但未循环结束
            last_signal = False
            return
        if trend_idx is None:
            last_signal = True # 没有开仓，但是已经进入过趋势线
            
        if leave_signal: # 执行开仓操作
            current_time = time[start_index]
        
        
        
        
        
        
        
        
        
        
