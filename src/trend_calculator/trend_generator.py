from sortedcontainers import SortedList
from src.trend_calculator.trend_process import initial_single_slope, calculate_trend
import time


class trend_generator():
    """
    回测用趋势数据生成器函数，能够动态检测外部对 data 的更新并调整循环次数。
    """
    def __init__(self):
        self.idx_price_high = 1
        self.idx_price_low = 3
        self.idx_time_high = 0
        self.idx_time_low = 2
        self.trend_high = [SortedList(key=lambda x: x[0])]
        self.trend_low = [SortedList(key=lambda x: x[0])]
        self.i = 1
        
    def initial_trend(self, data):
        """
        初始化趋势
        """
        for _ in range(len(data) - 1):
            next(self.__next__(data))
        
        return self.trend_high, self.trend_low
    
    def __next__(self, data):
        # 如果当前索引超出data中已有的元素，则等待新数据添加
        if self.i >= len(data):
            time.sleep(0.1)
            return

        self.trend_high.append(SortedList(key=lambda x: x[0]))
        self.trend_low.append(SortedList(key=lambda x: x[0]))
        
        # 当前回测数据：前 i+1 个数据
        backtest_data = data[:self.i+1]
        
        # 计算初始单点斜率
        initial_single_slope(backtest_data, trend=self.trend_high, idx_time=self.idx_time_high, idx_price=self.idx_price_high)
        initial_single_slope(backtest_data, trend=self.trend_low, idx_time=self.idx_time_low, idx_price=self.idx_price_low)
        
        # 原始趋势中因更新而被删除的趋势
        deleted_high = []
        deleted_low = []
        
        if self.i >= 2:
            calculate_trend(
                data=backtest_data, 
                trend_high=self.trend_high, 
                trend_low=self.trend_low, 
                start_idx=self.i, 
                deleted_high=deleted_high, 
                deleted_low=deleted_low
            )
            
        self.i += 1 
        yield self.trend_high, self.trend_low, deleted_high, deleted_low
        
        
