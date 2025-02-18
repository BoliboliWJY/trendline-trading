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
        self,
        data: np.ndarray,
        current_index: int,
        removed_items_high,
        removed_items_low,
    ):
        self.current_time = data[current_index][0]
        self.trend_high = removed_items_high
        self.trend_low = removed_items_low

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

    def evaluate_trade_signal(self, tick_price: float, trading_config: dict):
        """
        根据当前tick价格和趋势数据评估交易信号，分别检查高趋势和低趋势数据。
        """
        signals = {}

        # 检查高趋势数据，trend_type为'high'
        self._check_trend(
            trend_prices=self.trend_high_prices,
            state_flag="pre_state_high",
            signal_key="high_bounce",
            tick_price=tick_price,
            trading_config=trading_config,
            signals=signals,
            trend_type="high",
        )

        # 检查低趋势数据，trend_type为'low'
        self._check_trend(
            trend_prices=self.trend_low_prices,
            state_flag="pre_state_low",
            signal_key="low_bounce",
            tick_price=tick_price,
            trading_config=trading_config,
            signals=signals,
            trend_type="low",
        )

        return signals

    def _check_trend(
        self,
        trend_prices: list,
        state_flag: str,
        signal_key: str,
        tick_price: float,
        trading_config: dict,
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
          trading_config: 包含进入和退出阈值("enter_threshold"和"leave_threshold")的配置字典。
          signals: 用于记录交易信号的字典。
          trend_type: 指定当前趋势类型，取值 "high" 或 "low"。
        """
        if trend_prices:
            # 取当前最活跃的趋势阈值，使用列表最后一个元素
            active_threshold = trend_prices[-1]

            # 根据趋势类型计算与当前tick价格的距离
            if trend_type == "high":
                distance = 1 - tick_price / active_threshold
            else:  # "low"趋势
                distance = 1 - active_threshold / tick_price

            # 如果还未进入趋势区域，则判断是否达到进入阈值
            if not getattr(self, state_flag):
                if distance < trading_config.get("enter_threshold", 0.0002):
                    setattr(self, state_flag, True)
            else:
                # 已进入趋势区域，则判断是否离开或反向突破
                if distance > trading_config.get("leave_threshold", 0.0003):
                    signals[signal_key] = True
                    setattr(self, state_flag, False)  # 重置状态
                    trend_prices.pop()  # 一旦触发信号后移除该阈值
                elif distance < 0:
                    # 当距离变为负，说明价格已经突破趋势阈值，触发break信号
                    break_signal_key = signal_key.replace("bounce", "break")
                    signals[break_signal_key] = True
                    setattr(self, state_flag, False)  # 重置状态
                    trend_prices.pop()
