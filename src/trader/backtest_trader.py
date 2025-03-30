import numpy as np
from src.utils import profile_method
from src.order_book.order_booker import OrderBook

class BacktestTrader:
    def __init__(self, data: np.ndarray, trend_config: dict, trading_config: dict, order_manager: OrderBook):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.order_manager = order_manager
        
        self.lock = False # 是否锁定信号
        self.base_trend_number = -1 # 初始化值
        
        self.latest_trend_high_number = 0
        self.past_trend_high_number = 0
        self.latest_trend_low_number = 0
        self.past_trend_low_number = 0
        # 开仓时对应的最新趋势数据
        self.past_last_trend_high = []
        self.past_last_trend_low = []
        # 多余被删除的趋势数量
        self.missing_trend_high_number = 0
        self.missing_trend_low_number = 0
        
        self.open_signals = {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]}
        self.close_signals = {"high_close":[], "low_close":[]}
        
        self.order_book = []
        self.paused = False # 是否暂停可视化更新
        
        self.trend_price_high = np.zeros(10)
        self.trend_price_low = np.zeros(10)
        self.update_trend_price_num_threshold = trading_config["update_trend_price_num_threshold"]
        self.high_potential_signal = False
        self.low_potential_signal = False
        
        self.high_signal_num = 0
        self.low_signal_num = 0
        
        self.high_idx = []
        self.low_idx = []
        
        
        
        
        
    def update_trend_price(self, data, trend_price:np.ndarray, base_trend_number:int, latest_trend_high:int, latest_trend_low:int, latest_deleted_trend_high_number:int, latest_deleted_trend_low_number:int):
        """
        更新趋势价格
        """
        self.paused = False
        self.high_potential_signal = False
        self.low_potential_signal = False
        self.data = data
        self.base_trend_number = base_trend_number
        self.last_trend_high = latest_trend_high
        self.last_trend_low = latest_trend_low
        self.latest_trend_high_number = len(self.last_trend_high)
        self.latest_trend_low_number = len(self.last_trend_low)
        self.latest_deleted_trend_high_number = latest_deleted_trend_high_number
        self.latest_deleted_trend_low_number = latest_deleted_trend_low_number
        self.trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        self.trend_price_low = trend_price["trend_price_low"][:, 1]
        
        self.compare_trend_price_number()
        
        self.judge_open_signal()
        
    def compare_trend_price_number(self):
        # 比较趋势价格数量
        
        self.missing_trend_high_number += self.latest_deleted_trend_high_number - self.latest_trend_high_number
        
        self.missing_trend_low_number += self.latest_deleted_trend_low_number - self.latest_trend_low_number
        
        
            
        if self.latest_trend_high_number > self.update_trend_price_num_threshold and self.latest_trend_high_number > self.past_trend_high_number:
            self.past_trend_high_number = self.latest_trend_high_number
            self.high_potential_signal = True
            self.missing_trend_high_number = 0
        else:
            self.high_potential_signal = False
            
        # 低点增加且大于阈值数量
        if self.latest_trend_low_number > self.update_trend_price_num_threshold and self.latest_trend_low_number > self.past_trend_low_number:
            self.past_trend_low_number = self.latest_trend_low_number
            self.low_potential_signal = True
            self.missing_trend_low_number = 0
        else:
            self.low_potential_signal = False
            
        if self.missing_trend_high_number > 0 and self.latest_trend_high_number > self.update_trend_price_num_threshold:
            # 出现额外删除的趋势线
            # self.latest_trend_high_number -= self.missing_trend_high_number
            self.high_potential_signal = True
            self.missing_trend_high_number = 0
            
        if self.missing_trend_low_number > 0 and self.latest_trend_low_number > self.update_trend_price_num_threshold:
            # self.latest_trend_low_number -= self.missing_trend_low_number
            self.low_potential_signal = True
            self.missing_trend_low_number = 0
    
    # def judge_open_signal(self):
    #     # 判断是否开仓
    #     if self.high_potential_signal:
    #         if len(self.trend_price_low) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_high[0] > self.trading_config["potential_profit"]:
    #             # 重置趋势低点数量
    #             self.past_trend_low_number = 0
                
    #             self.high_signal_num += 1
    #             self.low_signal_num = 0
    #     if self.low_potential_signal:
    #         if len(self.trend_price_high) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_low[0] > self.trading_config["potential_profit"]:
    #             # 重置趋势高点数量
    #             self.past_trend_high_number = 0

    #             self.low_signal_num += 1
    #             self.high_signal_num = 0
    #     # 确认开仓发生
    #     if self.high_signal_num > self.trading_config["open_times"]:
    #         self.open_signals["high_open"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4]])
            
    #         self.close_signals["high_close"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4] * (1 + self.trading_config["trailing_stop_pct"])])
            
    #         # 记录开仓时间
    #         self.high_idx.append(self.base_trend_number)

    #         self.paused = True
    #         self.high_signal_num = 0
            
    #         self.past_last_trend_high.append(self.last_trend_high.copy())

    #     if self.low_signal_num > self.trading_config["open_times"]:
    #         self.open_signals["low_open"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4]])
            
    #         self.close_signals["low_close"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4] * (1 - self.trading_config["trailing_stop_pct"])])
            
    #         # 记录开仓时间
    #         self.low_idx.append(self.base_trend_number)

    #         self.paused = True
    #         self.low_signal_num = 0
            
    #         self.past_last_trend_low.append(self.last_trend_low.copy())
    
    def judge_open_signal(self):
        # 反向开仓
        # 判断是否开仓
        if self.high_potential_signal:
            if len(self.trend_price_low) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_high[0] > self.trading_config["potential_profit"]:
                # 重置趋势低点数量
                self.past_trend_low_number = 0
                
                self.high_signal_num += 1
                self.low_signal_num = 0
        if self.low_potential_signal:
            if len(self.trend_price_high) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_low[0] > self.trading_config["potential_profit"]:
                # 重置趋势高点数量
                self.past_trend_high_number = 0

                self.low_signal_num += 1
                self.high_signal_num = 0
        # 确认开仓发生 - 实现反向开仓
        if self.high_signal_num > self.trading_config["open_times"]:
            # 原本是做空，现在改为做多
            self.open_signals["high_open"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4]])
            
            # 止损点位调整为下跌百分比
            self.close_signals["high_close"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4] * (1 - self.trading_config["trailing_stop_pct"])])
            
            # 记录开仓时间，但放入 low_idx 而非 high_idx
            self.low_idx.append(self.base_trend_number)  # 放入低点索引，在计算盈亏时会按做多处理

            self.paused = True
            self.high_signal_num = 0
            
            self.past_last_trend_high.append(self.last_trend_high.copy())

        if self.low_signal_num > self.trading_config["open_times"]:
            # 原本是做多，现在改为做空
            self.open_signals["low_open"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4]])
            
            # 止损点位调整为上涨百分比
            self.close_signals["low_close"].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4] * (1 + self.trading_config["trailing_stop_pct"])])
            
            # 记录开仓时间，但放入 high_idx 而非 low_idx
            self.high_idx.append(self.base_trend_number)  # 放入高点索引，在计算盈亏时会按做空处理

            self.paused = True
            self.low_signal_num = 0
            
            self.past_last_trend_low.append(self.last_trend_low.copy())

    

    def place_order(self, order_type:str, price:float, tick_time, stop_loss, profit):
        order_id = len(self.order_book) + 1
        order = {
            "order_id": order_id,
            "order_type": order_type,
            "price": price,
            "tick_time": tick_time,
            "status": "open",
            "stop_loss": stop_loss,
            "profit": profit,
        }
        self.order_book.append(order)
    
    def calculate_profit_loss(self):
        # 计算所有交易盈亏
        for high_idx in self.high_idx:
            # 找到所有可能的低点作为候选
            potential_end_indices = [x for x in self.low_idx if x > high_idx]
            if not potential_end_indices:
                end_idx = len(self.data) - 1
            else:
                # 初始化第一个候选点
                end_idx = potential_end_indices[0]
                hit_stop_loss = False
                # 检查第一个潜在点是否达到止损
                temp_profit = self.calculate_profit_loss_low(end_idx, min(end_idx + 1000, len(self.data) - 1), check_stop_loss=True)
                if temp_profit == None:
                    hit_stop_loss = True # 触发止损
                
                if hit_stop_loss and len(potential_end_indices) > 1:
                    for next_end_idx in potential_end_indices[1:]:
                        # 检查后续潜在点是否达到止损
                        temp_profit = self.calculate_profit_loss_high(high_idx, next_end_idx, check_profit=True)
                        if temp_profit is not None and temp_profit >= 0:
                            end_idx = next_end_idx
                            break
            
            profit = self.calculate_profit_loss_high(high_idx, end_idx)
            self.place_order("buy", self.data[high_idx, 4], self.data[high_idx, -1], 1, profit)
            
        for low_idx in self.low_idx:
            # 找到所有可能的高点作为候选
            potential_end_indices = [x for x in self.high_idx if x > low_idx]
            if not potential_end_indices:
                end_idx = len(self.data) - 1
            else:
                # 初始化第一个候选点
                end_idx = potential_end_indices[0]
                hit_stop_loss = False
                # 检查第一个潜在点是否达到止损
                temp_profit = self.calculate_profit_loss_high(end_idx, min(end_idx + 1000, len(self.data) - 1), check_stop_loss = True)
                if temp_profit == None:
                    hit_stop_loss = True # 触发止损
                
                if hit_stop_loss and len(potential_end_indices) > 1:
                    for next_end_idx in potential_end_indices[1:]:
                        # 检查后续潜在点是否达到止损
                        temp_profit = self.calculate_profit_loss_low(low_idx, next_end_idx, check_profit=True)
                        if temp_profit is not None and temp_profit >= 0:
                            end_idx = next_end_idx
                            break

            profit = self.calculate_profit_loss_low(low_idx, end_idx)
            self.place_order("sell", self.data[low_idx, 4], self.data[low_idx, -1], 1, profit)

                            
        
        # for high_idx in self.high_idx:
        #     end_idx = next((x for x in self.low_idx if x > high_idx), len(self.data)) - 1
        #     profit = self.calculate_profit_loss_high(high_idx, end_idx)
        #     self.place_order("buy", self.data[high_idx, 4], self.data[high_idx, -1], 1, profit)
        # for low_idx in self.low_idx:
        #     end_idx = next((x for x in self.high_idx if x > low_idx), len(self.data)) - 1
        #     profit = self.calculate_profit_loss_low(low_idx, end_idx)
        #     self.place_order("sell", self.data[low_idx, 4], self.data[low_idx, -1], 1, profit)
        
    
    def calculate_profit_loss_high(self, start_idx:int, end_idx:int, check_stop_loss = False, check_profit=False):
        # 入场价格（成本价）
        entry_price = self.data[start_idx, 4]
        # 记录最低价格，用于移动止损
        ideal_price = entry_price
        
        for i in range(end_idx - start_idx):
            idx = start_idx + i
            current_high = self.data[idx, 1]
            current_low = self.data[idx, 3]
            
            # 如果价格上涨超过理想价格的一定比例，触发止损
            if (current_high - ideal_price) / ideal_price > self.trading_config["trailing_stop_pct"]:
                # 做空止损，收益为负
                profit = (entry_price - current_high) / entry_price
                if check_stop_loss and profit < 0:
                    return None # 触发止损
                return profit
            
            ideal_price = min(ideal_price, current_low)
            
            # 检查是否已经可以不亏损退出
            if check_profit:
                current_profit = (entry_price - self.data[idx, 4]) / entry_price
                if current_profit >= 0:
                    return current_profit
        
        return (entry_price - self.data[end_idx, 4]) / entry_price

    def calculate_profit_loss_low(self, start_idx:int, end_idx:int, check_stop_loss=False, check_profit=False):
        entry_price = self.data[start_idx, 4]  # 入场价格（成本价）
        # 记录最高价格，用于移动止损
        ideal_price = entry_price
        
        for i in range(end_idx - start_idx):
            idx = start_idx + i
            current_high = self.data[idx, 1]
            current_low = self.data[idx, 3]
            
            # 如果价格下跌超过理想价格的一定比例，触发止损
            if (ideal_price - current_low) / ideal_price > self.trading_config["trailing_stop_pct"]:
                # 做多止损，收益为负
                profit = (current_low - entry_price) / entry_price
                if check_stop_loss and profit < 0:
                    return None  # 返回止损点的收益率，表示触发了止损
                return profit
            
            ideal_price = max(ideal_price, current_high)
            
            # 检查是否已经可以不亏损退出
            if check_profit:
                current_profit = (self.data[idx, 4] - entry_price) / entry_price
                if current_profit >= 0:
                    return current_profit
            
        return (self.data[end_idx, 4] - entry_price) / entry_price
            
            
    
    # def calculate_profit_loss_high(self, start_idx:int, end_idx:int):
    #     # 入场价格（成本价）
    #     entry_price = self.data[start_idx, 4]
    #     # 记录最低价格，用于移动止损
    #     ideal_price = entry_price

    #     for i in range(end_idx - start_idx):
    #         idx = start_idx + i
    #         current_high = self.data[idx, 1]
    #         current_low = self.data[idx, 3]
            
    #         # 如果价格上涨超过理想价格的一定比例，触发止损
    #         if (current_high - ideal_price) / ideal_price > self.trading_config["trailing_stop_pct"]:
    #             # 做空止损，收益为负
    #             return (entry_price - current_high) / entry_price
            
    #         ideal_price = min(ideal_price, current_low)

    #     return (entry_price - self.data[end_idx, 4]) / entry_price

    # def calculate_profit_loss_low(self, start_idx:int, end_idx:int):
    #     entry_price = self.data[start_idx, 4]  # 入场价格（成本价）
    #     # 记录最高价格，用于移动止损
    #     ideal_price = entry_price
        
    #     for i in range(end_idx - start_idx):
    #         idx = start_idx + i
    #         current_high = self.data[idx, 1]
    #         current_low = self.data[idx, 3]
            
    #         # 如果价格下跌超过理想价格的一定比例，触发止损
    #         if (ideal_price - current_low) / ideal_price > self.trading_config["trailing_stop_pct"]:
    #             # 做多止损，收益为负
    #             return (current_low - entry_price) / entry_price
            
    #         ideal_price = max(ideal_price, current_high)
            
    #     return (self.data[end_idx, 4] - entry_price) / entry_price
            






    
        
