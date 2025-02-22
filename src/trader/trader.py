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
        self.signals = {}
        self.close_signals = {}
        # 是否触发break信号
        self.break_signal = {
            "high": False,
            "low": False,
        }
        # 记录初始价格，避免被多次出现的买点价格覆盖
        self.initial_price_low = 0
        self.initial_times_low = 0
        self.initial_price_high = 0
        self.initial_times_high = 0
        # 当前仓位订单信息
        self.book_order = {}
        # 历史仓位订单信息
        self.history_order = {}

        self.fee = trading_config.get("fee", 0.001)

        self.trailing = False # 是否开启移动止损

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
                self.break_signal[trend_type] = True
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
                    # 发出开仓信号
                    signals[signal_key] = True
                    bounce_signal_key = signal_key.replace("bounce", "tick_price")
                    signals.setdefault(bounce_signal_key, []).append(tick_price)
                    # 记录开仓时间
                    open_signal_key = signal_key.replace("bounce", "open")
                    signals.setdefault(open_signal_key, []).append(tick_timestamp)
                    setattr(self, state_flag, False)  # 重置状态
                    # 记录开仓信号
                    open_signal_key = signal_key.replace("bounce", "open")
                    self.book_order.setdefault(open_signal_key, []).append(
                        [tick_timestamp, tick_price]
                    )
                    # 记录初始价格
                    if trend_type == "high":
                        if not self.initial_price_high:
                            self.initial_price_high = tick_price
                            self.initial_times_high = tick_timestamp
                    else:
                        if not self.initial_price_low:
                            self.initial_price_low = tick_price
                            self.initial_times_low = tick_timestamp
                    self.break_signal[trend_type] = False  # 重置break信号

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
            print(
                "卖点开仓信号",
                signals.get("high_tick_price", "无卖点价格信息"),
                "时间为",
                datetime.datetime.fromtimestamp(tick_timestamp / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "趋势序号：",
                base_trend_number,
            )
        if signals.get("low_bounce", None) is not None:
            print(
                "买点开仓信号",
                signals.get("low_tick_price", "无买点价格信息"),
                "时间为",
                datetime.datetime.fromtimestamp(tick_timestamp / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )

    def monitor_close(self, tick_price: float, tick_timestamp: int, signals: dict, index: int):
        """
        监控平仓信号
        Args:
            tick_price: float, 当前tick价格
            tick_timestamp: int, 当前tick时间戳
            signals: dict, 交易信号，包含开仓信号和开仓价格还有开仓时间
        """
        # 处理高趋势单平仓
        # if "high_open" in self.book_order and self.book_order["high_open"]:
        #     stop_loss = 1 - self.book_order["high_open"][0][1] / tick_price >= self.trading_config.get("stop_loss", 0.008)

        #     if (
        #         stop_loss
        #         or self.break_signal["high"]
        #         or self.signals.get("low_bounce", None) is not None
        #         or 1 - tick_price / self.initial_price_high >= 0.001 # 大于2倍手续费就收手
        #     ):
        #         # 当大于止损率，或者触发break信号，或者出现新的低点反弹信号，需要平仓
        #         profit = 1 - tick_price / self.initial_price_high - self.fee
        #         history_record = [
        #             self.initial_times_high,
        #             self.initial_price_high,
        #             tick_timestamp,
        #             tick_price,
        #             profit,
        #         ]
        #         self.history_order.setdefault("high_order", []).append(history_record)
        #         print(
        #             "卖点平仓信号",
        #             history_record[-2],
        #             "时间为",
        #             datetime.datetime.fromtimestamp(history_record[-3] / 1000).strftime(
        #                 "%Y-%m-%d %H:%M:%S"
        #             ),
        #             "趋势序号：", index
        #         )
        #         # 生成平仓信号
        #         close_signal_key = "high_close"
        #         self.close_signals[close_signal_key] = True
        #         tick_price_key = close_signal_key.replace("close", "tick_price")
        #         self.close_signals.setdefault(tick_price_key, []).append(tick_price)
        #         timestamp_key = close_signal_key.replace("close", "timestamp")
        #         self.close_signals.setdefault(timestamp_key, []).append(tick_timestamp)

        #         self.book_order["high_open"] = []

        #         self.initial_price_high = 0
        #         self.initial_times_high = 0
        # else:
        #     # 若没有有效的高趋势订单，则无需处理
        #     pass

        # # 处理低趋势单平仓
        # if "low_open" in self.book_order and self.book_order["low_open"]:
        #     stop_loss = 1 - tick_price / self.book_order["low_open"][0][1] >= self.trading_config.get("stop_loss", 0.008)
        #     if (
        #         stop_loss
        #         or self.break_signal["low"]
        #         or self.signals.get("high_bounce", None) is not None
        #         or 1 - self.initial_price_low / tick_price >= 0.001 # 大于当前止盈手续费就收手
        #     ):
        #         # 当大于止损率，或者触发break信号，或者出现新的高点反弹信号，需要平仓
        #         profit = 1 - self.initial_price_low / tick_price - self.fee
        #         history_record = [
        #             self.initial_times_low,
        #             self.initial_price_low,
        #             tick_timestamp,
        #             tick_price,
        #             profit,
        #         ]
        #         self.history_order.setdefault("low_order", []).append(history_record)
        #         print(
        #             "买点平仓信号",
        #             history_record[-2],
        #             "时间为",
        #             datetime.datetime.fromtimestamp(history_record[-3] / 1000).strftime(
        #                 "%Y-%m-%d %H:%M:%S"
        #             ),
        #             "趋势序号：", index
        #         )

        #         # 生成平仓信号
        #         close_signal_key = "low_close"
        #         self.close_signals[close_signal_key] = True
        #         tick_price_key = close_signal_key.replace("close", "tick_price")
        #         self.close_signals.setdefault(tick_price_key, []).append(tick_price)
        #         timestamp_key = close_signal_key.replace("close", "timestamp")
        #         self.close_signals.setdefault(timestamp_key, []).append(tick_timestamp)

        #         self.book_order["low_open"] = []

        #         self.initial_price_low = 0
        #         self.initial_times_low = 0
        # else:
        #     # 若没有有效的低趋势订单，则无需处理
        #     pass

        # self.book_order["high_open"][0][1]为开仓价格

        if "high_open" in self.book_order and self.book_order["high_open"]: # 有高趋势订单
            if not self.trailing:
                stop_loss = (1 - self.book_order["high_open"][0][1] / tick_price) >= self.trading_config.get("stop_loss", 0.008) # 固定止损，防爆仓
            else:
                stop_loss = 1 - tick_price / self.book_order["high_open"][0][1] <= self.trading_config.get("trailing_stop_loss", 0.001) # 移动止损
            # stop_loss = 1 - self.book_order["high_open"][0][1] / tick_price >= self.trading_config.get("stop_loss", 0.008)
            # 如果利润空间大于阈值，则开启移动止损
            if 1 - tick_price / self.book_order["high_open"][0][1] >= self.trading_config.get("trailing_profit_threshold", 0.002):
                self.trailing = True # 开启移动止损
                

            if (
                stop_loss
                or self.break_signal["high"] # 同向方向触发break信号（支撑被打破，可能造成更多损失）
            ):
                profit = 1 - tick_price / self.book_order["high_open"][0][1] - self.fee
                history_record = [
                    self.book_order["high_open"][0][0],
                    self.book_order["high_open"][0][1],
                    tick_timestamp,
                    tick_price,
                    profit,
                ]
                self.history_order.setdefault("high_order", []).append(history_record)
                print(
                    "卖点平仓信号",
                    history_record[-2],
                    "时间为",
                    datetime.datetime.fromtimestamp(history_record[-3] / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "趋势序号：", index,
                    "利润：", history_record[-1]
                )

                # 生成平仓信号
                close_signal_key = "high_close"
                self.close_signals[close_signal_key] = True
                tick_price_key = close_signal_key.replace("close", "tick_price")
                self.close_signals.setdefault(tick_price_key, []).append(tick_price)
                timestamp_key = close_signal_key.replace("close", "timestamp")
                self.close_signals.setdefault(timestamp_key, []).append(tick_timestamp)

                self.book_order["high_open"] = []

                self.initial_price_high = 0
                self.initial_times_high = 0

                self.trailing = False # 关闭移动止损
        else:
            # 若没有有效的订单，则无需处理
            pass

        if "low_open" in self.book_order and self.book_order["low_open"]: # 有低趋势订单
            if not self.trailing:
                stop_loss = (1 - tick_price / self.book_order["low_open"][0][1]) >= self.trading_config.get("stop_loss", 0.008) # 固定止损，防爆仓
            else:
                stop_loss = 1 - self.book_order["low_open"][0][1] / tick_price <= self.trading_config.get("trailing_stop_loss", 0.001) # 移动止损
            # 如果利润空间大于阈值，则开启移动止损
            if 1 - self.book_order["low_open"][0][1] / tick_price >= self.trading_config.get("trailing_profit_threshold", 0.002):
                self.trailing = True # 开启移动止损
                
            if(
                stop_loss
                or self.break_signal["low"] # 同向方向触发break信号（阻力被打破，可能造成更多损失）
            ):
                profit = 1 - self.book_order["low_open"][0][1] / tick_price - self.fee
                history_record = [
                    self.book_order["low_open"][0][0],
                    self.book_order["low_open"][0][1],
                    tick_timestamp,
                    tick_price,
                    profit,
                ]
                self.history_order.setdefault("low_order", []).append(history_record)
                print(
                    "买点平仓信号",
                    history_record[-2],
                    "时间为",
                    datetime.datetime.fromtimestamp(history_record[-3] / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "趋势序号：", index,
                    "利润：", history_record[-1]
                )

                # 生成平仓信号
                close_signal_key = "low_close"
                self.close_signals[close_signal_key] = True
                tick_price_key = close_signal_key.replace("close", "tick_price")
                self.close_signals.setdefault(tick_price_key, []).append(tick_price)
                timestamp_key = close_signal_key.replace("close", "timestamp")

                self.book_order["low_open"] = []

                self.initial_price_low = 0
                self.initial_times_low = 0

                self.trailing = False # 关闭移动止损
        else:
            # 若没有有效的订单，则无需处理
            pass
