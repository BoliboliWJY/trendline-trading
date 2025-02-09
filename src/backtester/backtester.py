import numpy as np
from src.trading_strategy import TradingStrategy

class Backtester:
    """
    Backtester: Handles trading strategy execution for backtesting.
    """
    def __init__(self, data, basic_config, trend_config, trading_config):
        self.data = data
        self.basic_config = basic_config
        self.trend_config = trend_config
        self.trading_config = trading_config

        # Trade records; these can be later used for further analysis.
        self.record_trade = []
        self.result_trade = []
        self.total_result = [0, 0]

    def run_strategy(self, trend_high, trend_low):
        """
        Executes the trading strategy against the provided trend data.
        """
        # Create/update trade records using the TradingStrategy class.
        self.record_trade = TradingStrategy(
            data=self.data,
            trend_high=trend_high,
            trend_low=trend_low,
            basic_config=self.basic_config,
            trend_config=self.trend_config,
            trading_config=self.trading_config,
            record_trade=self.record_trade,
            result_trade=self.result_trade,
            total_result=self.total_result
        ).return_data()
        return self.record_trade

    def get_trade_markers(self):
        """
        Returns the x, y coordinates for plotting trade signals (long and short).
        """
        long_x, long_y, short_x, short_y = [], [], [], []
        for trade in self.record_trade:
            # Assuming that trade[-1] holds either 'long' or 'short'.
            if trade[-1] == 'long':
                long_x.append(trade[0][0])
                long_y.append(trade[0][1])
            elif trade[-1] == 'short':
                short_x.append(trade[0][0])
                short_y.append(trade[0][2])
        return long_x, long_y, short_x, short_y