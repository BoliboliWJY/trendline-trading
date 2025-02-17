import numpy as np


class Trader:
    """
    交易员类，用于处理交易逻辑
    输入tick数据，当前状态下的trend数据用于判断开平仓策略
    """

    def __init__(self):
        self.tick_price = None
        self.trend_high = None
        self.trend_low = None

        # 记录是否进入趋势阈值范围内
        self.pre_state_high = False
        self.pre_state_low = False

    def get_trend_data(
        self, data: np.ndarray, current_index: int, trend_high, trend_low
    ):
        self.current_time = data[current_index][0]
        self.trend_high = trend_high
        self.trend_low = trend_low

        # 利用向量化计算 trend_high_prices
        valid_trend_high = [
            pair for sublist in trend_high if sublist for pair in sublist if pair
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

        # 利用向量化计算 trend_low_prices
        valid_trend_low = [
            pair for sublist in trend_low if sublist for pair in sublist if pair
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

    def evaluate_trade_signal(self, tick_price: float, trading_config: dict):
        """
        评估交易信号
        """
        signals = {}

        if self.trend_high_prices:
            high_level = self.trend_high_prices[-1]
            distance_high = abs(tick_price - high_level)

            if not self.pre_state_high:
                if distance_high < trading_config.get("enter_threshold", 0.0002):
                    self.pre_state_high = True
            else:
                if distance_high > trading_config.get("leave_threshold", 0.0003):
                    signals["high_bounce"] = True
                    self.pre_state_high = False  # 重置状态
                    self.trend_high_prices.pop(0)

        if self.trend_low_prices:
            low_level = self.trend_low_prices[-1]
            distance_low = abs(tick_price - low_level)

            if not self.pre_state_low:
                if distance_low < trading_config.get("enter_threshold", 0.0002):
                    self.pre_state_low = True
            else:
                if distance_low > trading_config.get("leave_threshold", 0.0003):
                    signals["low_bounce"] = True
                    self.pre_state_low = False  # 重置状态
                    self.trend_low_prices.pop(0)

        return signals
