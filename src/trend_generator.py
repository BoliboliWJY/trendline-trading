from sortedcontainers import SortedList

def backtest_calculate_trend_generator(data, initial_single_slope, calculate_trend):
    """
    回测用趋势数据生成器函数
    """
    idx_high = 1
    idx_low = 2
    trend_high = [SortedList(key=lambda x: x[0])]
    trend_low = [SortedList(key=lambda x: x[0])]

    for i in range(1, len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        trend_low.append(SortedList(key=lambda x: x[0]))
        
        # 当前回测数据：前 i+1 个数据
        backtest_data = data[:i+1]
        
        # 计算初始单点斜率
        initial_single_slope(backtest_data, trend=trend_high, idx=idx_high)
        initial_single_slope(backtest_data, trend=trend_low, idx=idx_low)
        
        deleted_high = set()
        deleted_low = set()
        
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