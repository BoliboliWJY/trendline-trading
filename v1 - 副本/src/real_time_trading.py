import time
from binance import Client
from src.data_loader import load_or_fetch_data
from src.latest_data.new_price import get_current_price
from src.latest_data.latest_klines import get_latest_klines
from src.trading_strategy import TradingStrategy
from src.config_manager import load_basic_config

class RealTimeTrading:
    """
    A realtime trading class that periodically fetches the latest data,
    updates our trend calculations/strategy, and executes the trading logic.
    """
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.coin_type = config['coin_type']
        self.interval = config['interval']
        self.total_length = config['total_length']
        
        # Load initial historical data.
        # This will be used to build an initial trend snapshot.
        self.data, self.type_data = load_or_fetch_data(client, self.coin_type, self.interval, self.total_length)
        
        # For realtime trading, you might compute/update trends on the fly,
        # so here we start with empty trend lists.
        self.trend_high = []  
        self.trend_low = []
        
        # Trade records storage.
        self.record_trade = []
        self.result_trade = []
        self.total_result = [0, 0]
        
        # Create an instance of our existing TradingStrategy.
        # Note: In our current TradingStrategy design, the trend arrays are used
        # to compute points. In realtime, you might want to recalc them from scratch
        # or adapt the strategy to use streaming data.
        self.strategy = TradingStrategy(
            self.data, self.trend_high, self.trend_low,
            self.config, config.get('trend_config', {}), config.get('trading_config', {}),
            self.record_trade, self.result_trade, self.total_result
        )
        
    def fetch_lastest_data(self):
        