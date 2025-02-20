import numpy as np
import datetime

class Trader:
    """
    交易员类，用于处理交易逻辑
    输入tick数据，当前状态下的trend数据用于判断开平仓策略
    """

    def __init__(self, trading_config: dict):
        self.tick_price = None
        self.trend_high = None
        self.trend_low = None

        # 记录是否进入趋势阈值范围内
        self.pre_state_high = False
        self.pre_state_low = False
        
        self.trading_config = trading_config

        # 记录交易信号
        self.signals = {"tick_price": []}
        # self.signals = {
        #     "high_bounce": False,
        #     "low_bounce": False,
        #     "high_tick_price": [],
        #     "low_tick_price": [],
        # }

    def get_trend_data(
        self,
        data: np.ndarray,
        current_index: int,
        removed_items_high,
        removed_items_low,
    ):
        self.current_time = data[current_index][0]
        self.trend_high = removed_items_high
        self.trend_low = removed_items_low

        # 记录是否逼近趋势线
        hit_line = False
        
        # 利用向量化计算 trend_high_prices
        valid_trend_high = [
            pair for sublist in self.trend_high if sublist for pair in sublist if pair
        ]
        if valid_trend_high:
            slopes = [pair[0] for pair in valid_trend_high]
            start_indices = [pair[1] for pair in valid_trend_high]
            base_times = data[start_indices, 0]
            base_prices = data[start_indices, 1]
            prices = (base_prices + slopes * (self.current_time - base_times)).tolist()
            self.trend_high_prices = sorted(prices, reverse=True)
        else:
            self.trend_high_prices = []
            
        nearest_tick_price = self.trend_high_prices[-1]
        current_k_line_price = data[current_index, 1]
        distance = 1 - current_k_line_price / nearest_tick_price
        if distance < self.trading_config.get("enter_threshold", 0.0003):
            hit_line = True

        # 利用向量化计算 trend_low_prices
        valid_trend_low = [
            pair for sublist in self.trend_low if sublist for pair in sublist if pair
        ]
        if valid_trend_low:
            slopes = [pair[0] for pair in valid_trend_low]
            start_indices = [pair[1] for pair in valid_trend_low]
            base_times = data[start_indices, 0]
            base_prices = data[start_indices, 2]
            prices = (base_prices + slopes * (self.current_time - base_times)).tolist()
            self.trend_low_prices = sorted(prices)
        else:
            self.trend_low_prices = []
            
        nearest_tick_price = self.trend_low_prices[-1]
        current_k_line_price = data[current_index, 2]
        distance = 1 - nearest_tick_price / current_k_line_price
        if distance < self.trading_config.get("enter_threshold", 0.0002):
            hit_line = True
            
        return hit_line
            
        

    def evaluate_trade_signal(self, tick_price: float, tick_timestamp: int):
        """
        根据当前tick价格和趋势数据评估交易信号，分别检查高趋势和低趋势数据。
        """

        # 检查高趋势数据，trend_type为'high'
        self._check_trend(
            trend_prices=self.trend_high_prices,
            reverse_prices=self.trend_low_prices,
            state_flag="pre_state_high",
            signal_key="high_bounce",
            tick_price=tick_price,
            tick_timestamp=tick_timestamp,
            signals=self.signals,
            trend_type="high",
        )

        # 检查低趋势数据，trend_type为'low'
        self._check_trend(
            trend_prices=self.trend_low_prices,
            reverse_prices=self.trend_high_prices,
            state_flag="pre_state_low",
            signal_key="low_bounce",
            tick_price=tick_price,
            tick_timestamp=tick_timestamp,
            signals=self.signals,
            trend_type="low",
        )

        return self.signals

    def _check_trend(
        self,
        trend_prices: list,
        reverse_prices: list,
        state_flag: str,
        signal_key: str,
        tick_price: float,
        tick_timestamp: int,
        signals: dict,
        trend_type: str,
    ):
        """
        检查当前趋势数据，更新交易状态并判断是否发出信号。

        参数:
          trend_prices: 当前趋势的阈值价格列表。
          state_flag: 状态属性名称（例如 "pre_state_high" 或 "pre_state_low"）。
          signal_key: 信号字典中对应的键（例如 "high_bounce" 或 "low_bounce"）。
          tick_price: 当前tick价格。
          tick_timestamp: 当前tick时间戳。
          signals: 用于记录交易信号的字典。
          trend_type: 指定当前趋势类型，取值 "high" 或 "low"。
        """
        if trend_prices:
            # 取当前最活跃的趋势阈值，使用列表最后一个元素
            active_threshold = trend_prices[-1]
            reverse_active_threshold = reverse_prices[-1]
            # 根据趋势类型计算与当前tick价格的距离
            if trend_type == "high":
                distance = 1 - tick_price / active_threshold
                reverse_distance = 1 - reverse_active_threshold / tick_price
            else:  # "low"趋势
                distance = 1 - active_threshold / tick_price
                reverse_distance = 1 - tick_price / reverse_active_threshold

            # 如果当前的趋势线已经被突破，则需要弹出并重新计算新的阈值数据点
            while trend_prices and distance < 0:
                setattr(self, state_flag, False)  # 重置状态
                trend_prices.pop()
                if trend_prices:
                    active_threshold = trend_prices[-1]
                    if trend_type == "high":
                        distance = 1 - tick_price / active_threshold
                        reverse_distance = 1 - reverse_active_threshold / tick_price
                    else:  # "low"趋势
                        distance = 1 - active_threshold / tick_price
                        reverse_distance = 1 - tick_price / reverse_active_threshold
                else:
                    break
            # 先判断是否存在潜在利润空间
            min_profit = self.trading_config.get("potential_profit", 0.01)
            has_profit_potential = reverse_distance > min_profit

            # 如果还未进入趋势区域，先评估是否值得进场
            if not getattr(self, state_flag):
                # 只有在有足够利润空间的情况下才考虑进场
                if has_profit_potential and distance < self.trading_config.get(
                    "enter_threshold", 0.0002
                ):
                    setattr(self, state_flag, True)
            else:
                # 已进入趋势区域，则判断是否离开或反向突破
                if distance > self.trading_config.get("leave_threshold", 0.0003):
                    signals[signal_key] = True
                    bounce_signal_key = signal_key.replace("bounce", "tick_price")
                    signals.setdefault(bounce_signal_key, []).append(tick_price)
                    signals.setdefault("tick_timestamp", []).append(tick_timestamp)
                    setattr(self, state_flag, False)  # 重置状态
                    # trend_prices.pop()  # 一旦触发信号后移除该阈值
                elif distance < 0:
                    # 当距离变为负，说明价格已经突破趋势阈值，触发break信号
                    # break_signal_key = signal_key.replace("bounce", "break")
                    # signals[break_signal_key] = True
                    # signals["tick_price"].append(tick_price)
                    # setattr(self, state_flag, False)  # 重置状态
                    trend_prices.pop()

    def notification_open(self, data, base_trend_number, signals):
        """
        通知开仓信号
        """
        tick_timestamp = data[base_trend_number, 0]
        tick_time = datetime.datetime.fromtimestamp(tick_timestamp / 1000)
        if signals.get("high_bounce", None) is not None:
            print("K线时间：", tick_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("出现卖点", signals.get("high_tick_price", "无卖点价格信息"))
        if signals.get("low_bounce", None) is not None:
            print("K线时间：", tick_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("出现买点", signals.get("low_tick_price", "无买点价格信息"))
            
    
    def monitor_close(self, tick_price: float, tick_timestamp: int, signals: dict):
        """
        监控平仓信号
        Args:
            tick_price: float, 当前tick价格
            tick_timestamp: int, 当前tick时间戳
            signals: dict, 交易信号，包含开仓信号和开仓价格还有开仓时间
        """
        # TODO 趋势线在tick价格可由self.trend_high_prices和self.trend_low_prices获取
        # TODO 除了止损，止盈信号可以由新的反向反弹信号，或者与止损方向最近的趋势价格相比较，突破了则进行平仓
        if signals.get("high_bounce", None) is not None:
            pass
        else:
            
            
            
        if signals.get("low_bounce", None) is not None:
            pass
        
        
        
