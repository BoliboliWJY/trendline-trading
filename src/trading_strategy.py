class TradingStrategy:
    def __init__(self, data, trend_high, trend_low, config):
        self.data = data
        self.trend_high = trend_high
        self.trend_low = trend_low
        self.config = config
        
    def evaluate(self, trend_high, trend_low):
        pass
    
    def buy(self):
        pass
    
    def sell(self):
        pass