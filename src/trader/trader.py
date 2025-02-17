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
            self.trend_high_prices = (
                base_prices + slopes * (self.current_time - base_times)
            ).tolist()
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
            base_prices = data[start_indices, 1]
            self.trend_low_prices = (
                base_prices + slopes * (self.current_time - base_times)
            ).tolist()
        else:
            self.trend_low_prices = []
