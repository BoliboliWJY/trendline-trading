def initial_slope(data, trend, idx):
    """初始化斜率数组

    Args:
        data (np.list): 数据，0为时间，1为最高价，2为最底下
        trend (sortedlist): 保存斜率数据
        idx (int): 选择第1或2列数据
    """
    for i in range(1, len(data)):
        j = i-1 
        current_time, current_value = data[i,[0,idx]]
        prev_time, prev_value = data[j,[0,idx]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        trend[i].add((slope,j))
        
def update_trend_high(data, trend_high, current_idx, i, current_slope):
    index = trend_high[i].bisect_left((current_slope,))
    for j in range(index - 1, -1, -1):
        current_time, current_value = data[current_idx, [0, 1]]
        prev_index = trend_high[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 1]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_high[i][j]
        trend_high[current_idx].add((slope, prev_index))
        update_trend_high(data, trend_high, current_idx, prev_index, slope)

def update_trend_low(data, trend_low, current_idx, i, current_slope):
    index = trend_low[i].bisect_right((current_slope,))
    if len(trend_low[i]) == 0:
        return
    for j in range(len(trend_low[i])-1, index-1, -1):
        current_time, current_value = data[current_idx, [0, 2]]
        prev_index = trend_low[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 2]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_low[i][j]
        trend_low[current_idx].add((slope, prev_index))
        update_trend_low(data, trend_low, current_idx, prev_index, slope)
        
def initial_single_slope(data, trend, idx):
    """初始化斜率数组,但每次只初始化最后一个元素的斜率

    Args:
        data (np.list): 数据，0为时间，1为最高价，2为最底下
        trend (sortedlist): 保存斜率数据
        idx (int): 选择第1或2列数据
    """
    i = len(data)-1
    j = i - 1
    current_time, current_value = data[i,[0,idx]]
    prev_time, prev_value = data[j,[0,idx]]
    slope = (current_value - prev_value) / (current_time - prev_time)
    trend[i].add((slope,j))
    
from sortedcontainers import SortedList

def new_trend(data, update_trend_high, update_trend_low, initial_single_slope, trend_high, idx_high, trend_low, idx_low):
    for i in range(1,len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        initial_single_slope(data[:i+1], trend_high, idx_high)
        current_slope = trend_high[i][0][0]
        update_trend_high(data,trend_high = trend_high,current_idx = i, i = i - 1, current_slope=current_slope)
    
        trend_low.append(SortedList(key=lambda x: x[0]))  
        initial_single_slope(data[:i+1], trend_low, idx_low)
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low = trend_low, current_idx = i, i = i - 1, current_slope=current_slope)


def realtime_trend(data):
    trend_high = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_high = 1
    initial_slope(data, trend_high, idx_high)
    for i in range(2,len(data)):
        current_slope = trend_high[i][0][0]
        update_trend_high(data, trend_high=trend_high, current_idx=i, i=i - 1, 
current_slope=current_slope)
    trend_low = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_low = 2
    initial_slope(data, trend_low,idx_low)
    for i in range(2,len(data)):
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low=trend_low, current_idx=i, i=i - 1, 
current_slope=current_slope)