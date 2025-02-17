
class Trader:
    """
    交易员类，用于处理交易逻辑
    输入tick数据，当前状态下的trend数据用于判断开平仓策略
    """
    def __init__(self, main_trend:dict,tick_time:int):
        self.trend_high = main_trend["trend_high"]
        self.trend_low = main_trend["trend_low"]
        


