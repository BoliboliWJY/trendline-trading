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

def update_trend_high(data, trend_high, current_idx, i, current_slope, deleted_high):
    index = trend_high[i].bisect_left((current_slope,))
    for j in range(index - 1, -1, -1):
        current_time, current_value = data[current_idx, [0, 1]]
        prev_index = trend_high[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 1]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        deleted_high.append((i, list(trend_high[i][j])))
        del trend_high[i][j]
        trend_high[current_idx].add((slope, prev_index))
        update_trend_high(data, trend_high, current_idx, prev_index, slope, deleted_high)

def update_trend_low(data, trend_low, current_idx, i, current_slope, deleted_low):
    index = trend_low[i].bisect_right((current_slope,))
    if len(trend_low[i]) == 0:
        return
    for j in range(len(trend_low[i])-1, index-1, -1):
        current_time, current_value = data[current_idx, [0, 2]]
        prev_index = trend_low[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 2]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        deleted_low.append((i, list(trend_low[i][j])))
        del trend_low[i][j]
        trend_low[current_idx].add((slope, prev_index))
        update_trend_low(data, trend_low, current_idx, prev_index, slope, deleted_low)

def calculate_trend(data, trend_high, trend_low, start_idx, deleted_high, deleted_low):
    """calculate rise/fall trend, it shall calculate the length of the data

    Args:
        data (array): market data
        update_trend_high (func): updating rise trend
        update_trend_low (func): updating fall trend
        trend_high (array): rise trend data
        trend_low (array): fall trend data
        start_idx (int): the beginning index for calculating, for realtime, it should be 2
    """
    for i in range(start_idx,len(data)):
        # start_time = time.perf_counter()
    #process high_time
        current_slope = trend_high[i][0][0]
        update_trend_high(data, trend_high=trend_high, current_idx=i, i=i - 1, current_slope=current_slope, deleted_high=deleted_high)
    #process low_time
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low=trend_low, current_idx=i, i=i - 1, current_slope=current_slope, deleted_low=deleted_low)