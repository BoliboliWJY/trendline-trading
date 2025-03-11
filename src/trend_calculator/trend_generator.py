from sortedcontainers import SortedList
from src.trend_calculator.trend_process import initial_single_slope, calculate_trend

def backtest_calculate_trend_generator(data):
    """
    回测用趋势数据生成器函数
    """
    idx_price_high = 1
    idx_price_low = 3
    idx_time_high = 0
    idx_time_low = 2
    trend_high = [SortedList(key=lambda x: x[0])]
    trend_low = [SortedList(key=lambda x: x[0])]

    for i in range(1, len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        trend_low.append(SortedList(key=lambda x: x[0]))
        
        # 当前回测数据：前 i+1 个数据
        backtest_data = data[:i+1]
        
        # 计算初始单点斜率
        initial_single_slope(backtest_data, trend=trend_high, idx_time=idx_time_high, idx_price=idx_price_high)
        initial_single_slope(backtest_data, trend=trend_low, idx_time=idx_time_low, idx_price=idx_price_low)
        
        # 原始趋势中因更新而被删除的趋势，last_filtered_xx中会包含这些需要被趋势
        deleted_high = []
        deleted_low = []
        
        if i >= 2:
            calculate_trend(
                data=backtest_data, 
                trend_high=trend_high, 
                trend_low=trend_low, 
                start_idx=i, 
                deleted_high=deleted_high, 
                deleted_low=deleted_low
            )
        yield trend_high, trend_low, deleted_high, deleted_low 
