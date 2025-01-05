import numpy as np

def filter_trend(trend_high, trend_low, data, config):
    """
    过滤趋势
    """
    #最近delay个数据不显示
    trend_high = trend_high[:-config.get('delay', 10)]
    trend_low = trend_low[:-config.get('delay', 10)]
    
    # 斜率大小限制
    if config.get('filter_slope', False):
        trend_high = filter_by_slope(trend_high, threshold=config.get('slope_threshold', 1))
        trend_low = filter_by_slope(trend_low, threshold=config.get('slope_threshold', 1))

    # 最小生成间隔
    if config.get('filter_line_age', False):
        trend_high = filter_by_line_age(trend_high, min_age=config.get('min_line_age', 5))
        trend_low = filter_by_line_age(trend_low, min_age=config.get('min_line_age', 5))

    # 最小距离
    if config.get('filter_distance', False):
        interval = config.get('interval', '1000*1000')
        trend_high = filter_by_distance(trend_high, data, distance_threshold=config.get('distance_threshold', 10), interval=interval)
        trend_low = filter_by_distance(trend_low, data, distance_threshold=config.get('distance_threshold', 10), interval=interval)

    if config.get('filter_trending_line', False):
        trend_high = filter_by_trending_line(trend_high)
        trend_low = filter_by_trending_line(trend_low)
        
    return trend_high, trend_low 

def filter_by_slope(trend, threshold=1):
    """Filter trends based on the absolute slope threshold."""
    filtered_trend = [[] for _ in range(len(trend))]
    for i in range(1, len(trend)):
        for slope, j in trend[i]:
            if np.abs(slope) <= threshold:
                filtered_trend[i].append([slope, j])
    return filtered_trend

def filter_by_line_age(trend, min_age):
    """Filter trends based on the minimum age of a trend line."""
    filtered_trend = [[] for _ in range(len(trend))]
    for i in range(1, len(trend)):
        for slope, j in trend[i]:
            line_age = i - j  # Example calculation; adjust as needed
            if line_age >= min_age:
                filtered_trend[i].append([slope, j])
    return filtered_trend

def filter_by_distance(trend, data, distance_threshold, interval):
    """Filter trends based on the distance between trend lines."""
    filtered_trend = [[] for _ in range(len(trend))]
    for i in range(1, len(trend)):
        for slope, j in trend[i]:
            distance = (data[i, 0] - data[j, 0]) / interval
            if distance >= distance_threshold:
                filtered_trend[i].append([slope, j])
    return filtered_trend

def filter_by_trending_line(trend):
    """过滤处于趋势之中趋势数量小于2的线"""
    filtered_trend = [[] for _ in range(len(trend))]
    for i in range(1, len(trend)):
        for slope, j in trend[i]:
            if len(trend[j]) < 1:
                continue
            filtered_trend[i].append([slope, j])
    return filtered_trend
