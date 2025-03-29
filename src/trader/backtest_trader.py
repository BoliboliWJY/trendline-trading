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
        
        # 开仓下，初当前趋势以外所有过去趋势的价格
        self.diff_trend_high_price = []
        self.diff_trend_low_price = []
        self.open_trend_high_data = []
        self.open_trend_low_data = []
        
        
    def update_trend_price(self, data, trend_price:np.ndarray, latest_trend_tick_data:np.ndarray, base_trend_number:int, latest_deleted_trend_high_number:int, latest_deleted_trend_low_number:int,
    latest_trend_high_data, latest_trend_low_data):
        """
        更新趋势价格
        """
        self.paused = False
        self.high_potential_signal = False
        self.low_potential_signal = False
        self.data = data
        self.base_trend_number = base_trend_number
        self.latest_trend_high_number = len(latest_trend_high_data)
        self.latest_trend_low_number = len(latest_trend_low_data)
        self.latest_deleted_trend_high_number = latest_deleted_trend_high_number
        self.latest_deleted_trend_low_number = latest_deleted_trend_low_number
        self.trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        self.trend_price_low = trend_price["trend_price_low"][:, 1]
        self.latest_trend_price_high = latest_trend_tick_data["trend_price_high"][:, 1]
        self.latest_trend_price_low = latest_trend_tick_data["trend_price_low"][:, 1]
        
        self.latest_trend_high_data = latest_trend_high_data.copy()
        self.latest_trend_low_data = latest_trend_low_data.copy()

        
        self.compare_trend_price_number()
        
        self.judge_open_signal()
        
    def compare_trend_price_number(self):
        # 比较趋势价格数量
        self._process_trend_data("high")
        self._process_trend_data("low")
        
    def _process_trend_data(self, trend_type="high"):
        is_high = trend_type == "high"
        # 更新缺失的趋势数量
        if is_high:
            self.missing_trend_high_number += self.latest_deleted_trend_high_number - self.latest_trend_high_number
            latest_trend_number = self.latest_trend_high_number
            past_trend_number = self.past_trend_high_number
        else:
            self.missing_trend_low_number += self.latest_deleted_trend_low_number - self.latest_trend_low_number
            latest_trend_number = self.latest_trend_low_number
            past_trend_number = self.past_trend_low_number
        
        # 检查趋势数量是否超过阈值且增加
        if latest_trend_number > self.update_trend_price_num_threshold and latest_trend_number > past_trend_number:
            if is_high:
                self.past_trend_high_number = latest_trend_number
                self.high_potential_signal = True
                self.missing_trend_high_number = 0
            else:
                self.past_trend_low_number = latest_trend_number
                self.low_potential_signal = True
                self.missing_trend_low_number = 0
        else:
            if is_high:
                self.high_potential_signal = False
            else:
                self.low_potential_signal = False
            
        # 处理额外删除的趋势线
        if is_high:
            if self.missing_trend_high_number > 0 and latest_trend_number > self.update_trend_price_num_threshold:
                self.high_potential_signal = True
                self.missing_trend_high_number = 0
        else:
            if self.missing_trend_low_number > 0 and latest_trend_number > self.update_trend_price_num_threshold:
                self.low_potential_signal = True
                self.missing_trend_low_number = 0
                
    def judge_open_signal(self):
        self._process_open_signal("high")
        self._process_open_signal("low")
        
    def _process_open_signal(self, trend_type="high"):
        is_high = trend_type == "high"
        potential_signal = self.high_potential_signal if is_high else self.low_potential_signal
        
        if potential_signal:
            # 计算潜在利润
            profit_condition = False
            if is_high:
                if len(self.trend_price_low) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_high[0] > self.trading_config["potential_profit"]:
                    profit_condition = True
            else:
                if len(self.trend_price_high) == 0 or (self.trend_price_high[0] - self.trend_price_low[0]) / self.trend_price_low[0] > self.trading_config["potential_profit"]:
                    profit_condition = True
            
            if profit_condition:
                # 重置对应的趋势点数量
                if is_high:
                    self.past_trend_low_number = 0
                    self.high_signal_num += 1
                    self.low_signal_num = 0
                else:
                    self.past_trend_high_number = 0
                    self.low_signal_num += 1
                    self.high_signal_num = 0
                    
        # 确认开仓发生
        signal_num = self.high_signal_num if is_high else self.low_signal_num
        if signal_num > self.trading_config["open_times"]:
            signal_key = "high_open" if is_high else "low_open"
            close_key = "high_close" if is_high else "low_close"
            multiplier = (1 + self.trading_config["trailing_stop_pct"]) if is_high else (1 - self.trading_config["trailing_stop_pct"])
            
            self.open_signals[signal_key].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4]])
            self.close_signals[close_key].append([self.data[self.base_trend_number, -1], self.data[self.base_trend_number, 4] * multiplier])
            
            # 记录开仓时间
            if is_high:
                self.high_idx.append(self.base_trend_number)
                self.high_signal_num = 0
                self.diff_trend_high_price.append(self.get_difference_trend_prices("high"))
                self.open_trend_high_data.append(self.latest_trend_high_data)
            else:
                self.low_idx.append(self.base_trend_number)
                self.low_signal_num = 0
                self.diff_trend_low_price.append(self.get_difference_trend_prices("low"))
                self.open_trend_low_data.append(self.latest_trend_low_data)
                
            self.paused = True

    def calculate_profit_loss(self):
        # 计算所有交易盈亏
        window_size = self.trading_config["window_size"]
        idx = 0
        for high_idx in self.high_idx:
            profit = self.calculate_profit_high(high_idx, self.open_trend_high_data[idx], self.diff_trend_high_price[idx], window_size)
            idx += 1
            self.place_order("buy", self.data[high_idx, 4], self.data[high_idx, -1], 1, profit)
        idx = 0
        for low_idx in self.low_idx:
            profit = self.calculate_profit_low(low_idx, self.open_trend_low_data[idx], self.diff_trend_low_price[idx], window_size)
            idx += 1
            self.place_order("sell", self.data[low_idx, 4], self.data[low_idx, -1], 1, profit)
    
    def calculate_profit_high(self, base_trend_number, trend_high_data, trailing_trend_price, window_size):
        # 计算高点的盈利
        entry_price = self.data[base_trend_number, 4]
        len_trend = len(trend_high_data)
        len_trend //= 2
        trend_data = trend_high_data[max(0, len_trend - 1)]
        stop_loss = self.trading_config["trailing_stop_pct"] # 最大回撤
        max_loss = trailing_trend_price[0] if trailing_trend_price.size > 0 else entry_price * (1 + stop_loss)
        ideal_price = entry_price
        for i in range(window_size):
            current_idx = min(base_trend_number + i, len(self.data) - 1)
            # 当前高低价格数据
            current_high_price = self.data[current_idx, 1]
            current_low_price = self.data[current_idx, 3]
            
            if current_high_price > max_loss:
                profit = (entry_price - current_high_price) / entry_price
                return profit
            
            if current_low_price < ideal_price:
                ideal_price = current_low_price
            
            # trend_stop_loss = self.data[trend_data[1], 1] + trend_data[0] * (self.data[current_idx, -1] - self.data
            
            if current_high_price > ideal_price * (1 + stop_loss):
                profit = (entry_price - current_high_price) / entry_price
                return profit
            
            # if current_high_price > trend_stop_loss:
            #     profit = (entry_price - current_high_price) / entry_price
            #     return profit
        
        final_profit = (entry_price - self.data[min(base_trend_number + window_size - 1, len(self.data) - 1), 4]) / entry_price
        return final_profit
    
    
    
    
    def calculate_profit_low(self, base_trend_number, trend_low_data, trailing_trend_price, window_size):
        # 计算低点的盈利(做多)
        entry_price = self.data[base_trend_number, 4]
        len_trend = len(trend_low_data)
        len_trend //= 2
        trend_data = trend_low_data[max(0, len_trend - 1)]
        stop_loss = self.trading_config["trailing_stop_pct"]
        max_loss = trailing_trend_price[-1] if trailing_trend_price.size > 0 else entry_price * (1 - stop_loss)
        ideal_price = entry_price
        for i in range(window_size):
            current_idx = min(base_trend_number + i, len(self.data) - 1)
            # 当前高低价格数据
            current_high_price = self.data[current_idx, 1]
            current_low_price = self.data[current_idx, 3]
            
            if current_low_price < max_loss:
                profit = (current_low_price - entry_price) / entry_price
                return profit
            
            if current_high_price > ideal_price:
                ideal_price = current_high_price
            
            # trend_stop_loss = self.data[trend_data[1], 3] + trend_data[0] * (self.data[current_idx, -1] - self.data[trend_data[1], -1])
            
            if current_low_price < ideal_price * (1 - stop_loss):
                profit = (current_low_price - entry_price) / entry_price
                return profit

            # if current_low_price < trend_stop_loss:
            #     profit = (current_low_price - entry_price) / entry_price
            #     return profit
            
        final_profit = (self.data[min(base_trend_number + window_size - 1, len(self.data) - 1), 4] - entry_price) / entry_price
        return final_profit
        
        
    
    
    # def calculate_profit_loss(self):
    #     # 计算所有交易盈亏
    #     for high_idx in self.high_idx:
    #         # 找到所有可能的低点作为候选
    #         potential_end_indices = [x for x in self.low_idx if x > high_idx]
    #         if not potential_end_indices:
    #             end_idx = len(self.data) - 1
    #         else:
    #             # 初始化第一个候选点
    #             end_idx = potential_end_indices[0]
    #             hit_stop_loss = False
    #             # 检查第一个潜在点是否达到止损
    #             temp_profit = self.calculate_profit_loss_low(end_idx, min(end_idx + 1000, len(self.data) - 1), check_stop_loss=True)
    #             if temp_profit == None:
    #                 hit_stop_loss = True # 触发止损
                
    #             if hit_stop_loss and len(potential_end_indices) > 1:
    #                 for next_end_idx in potential_end_indices[1:]:
    #                     # 检查后续潜在点是否达到止损
    #                     temp_profit = self.calculate_profit_loss_high(high_idx, next_end_idx, check_profit=True)
    #                     if temp_profit is not None and temp_profit >= 0:
    #                         end_idx = next_end_idx
    #                         break
            
    #         profit = self.calculate_profit_loss_high(high_idx, end_idx)
    #         self.place_order("buy", self.data[high_idx, 4], self.data[high_idx, -1], 1, profit)
            
    #     for low_idx in self.low_idx:
    #         # 找到所有可能的高点作为候选
    #         potential_end_indices = [x for x in self.high_idx if x > low_idx]
    #         if not potential_end_indices:
    #             end_idx = len(self.data) - 1
    #         else:
    #             # 初始化第一个候选点
    #             end_idx = potential_end_indices[0]
    #             hit_stop_loss = False
    #             # 检查第一个潜在点是否达到止损
    #             temp_profit = self.calculate_profit_loss_high(end_idx, min(end_idx + 1000, len(self.data) - 1), check_stop_loss = True)
    #             if temp_profit == None:
    #                 hit_stop_loss = True # 触发止损
                
    #             if hit_stop_loss and len(potential_end_indices) > 1:
    #                 for next_end_idx in potential_end_indices[1:]:
    #                     # 检查后续潜在点是否达到止损
    #                     temp_profit = self.calculate_profit_loss_low(low_idx, next_end_idx, check_profit=True)
    #                     if temp_profit is not None and temp_profit >= 0:
    #                         end_idx = next_end_idx
    #                         break

    #         profit = self.calculate_profit_loss_low(low_idx, end_idx)
    #         self.place_order("sell", self.data[low_idx, 4], self.data[low_idx, -1], 1, profit)

                            
        
        # for high_idx in self.high_idx:
        #     end_idx = next((x for x in self.low_idx if x > high_idx), len(self.data)) - 1
        #     profit = self.calculate_profit_loss_high(high_idx, end_idx)
        #     self.place_order("buy", self.data[high_idx, 4], self.data[high_idx, -1], 1, profit)
        # for low_idx in self.low_idx:
        #     end_idx = next((x for x in self.high_idx if x > low_idx), len(self.data)) - 1
        #     profit = self.calculate_profit_loss_low(low_idx, end_idx)
        #     self.place_order("sell", self.data[low_idx, 4], self.data[low_idx, -1], 1, profit)
        
    
    # def calculate_profit_loss_high(self, start_idx:int, end_idx:int, check_stop_loss = False, check_profit=False):
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
    #             profit = (entry_price - current_high) / entry_price
    #             if check_stop_loss and profit < 0:
    #                 return None # 触发止损
    #             return profit
            
    #         ideal_price = min(ideal_price, current_low)
            
    #         # 检查是否已经可以不亏损退出
    #         if check_profit:
    #             current_profit = (entry_price - self.data[idx, 4]) / entry_price
    #             if current_profit >= 0:
    #                 return current_profit
        
    #     return (entry_price - self.data[end_idx, 4]) / entry_price

    # def calculate_profit_loss_low(self, start_idx:int, end_idx:int, check_stop_loss=False, check_profit=False):
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
    #             profit = (current_low - entry_price) / entry_price
    #             if check_stop_loss and profit < 0:
    #                 return None  # 返回止损点的收益率，表示触发了止损
    #             return profit
            
    #         ideal_price = max(ideal_price, current_high)
            
    #         # 检查是否已经可以不亏损退出
    #         if check_profit:
    #             current_profit = (self.data[idx, 4] - entry_price) / entry_price
    #             if current_profit >= 0:
    #                 return current_profit
            
    #     return (self.data[end_idx, 4] - entry_price) / entry_price
            
            
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


    def get_difference_trend_prices(self, type_str="low"):
        """
        获取存在于完整趋势点中但不存在于最新趋势点中的数据

        参数:
            type_str: 'high' 或 'low'，表示要处理高点还是低点

        返回:
            numpy数组: 差集结果
        """
        if type_str == "high":
            return np.setdiff1d(self.trend_price_high, self.    latest_trend_price_high)
        elif type_str == "low":
            return np.setdiff1d(self.trend_price_low, self. latest_trend_price_low)
        else:
            raise ValueError("type_str 必须是 'high' 或 'low'")
    
    


    
        
