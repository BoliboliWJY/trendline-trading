from src.filter.filters import filter_trend_initial

def compute_initial_trends(current_data, trend_generator, data, trend_config, last_filtered_high, last_filtered_low):
    """
    计算初始趋势
    这个方法也可以用于实时交易初始化
    Args:
        current_data: 当前数据
        trend_generator: 趋势生成器
        data: 数据
        trend_config: 趋势配置
        last_filtered_high: 最后过滤的高趋势数据
        last_filtered_low: 最后过滤的低趋势数据
    Returns:
        trend_high, trend_low: 完整的趋势数据
        last_filtered_high, last_filtered_low: 最后过滤的趋势数据
    """
    try:
        for _ in range(len(current_data) - 1):

            trend_high, trend_low, deleted_high, deleted_low = next(
                trend_generator
            )
    except StopIteration:
        trend_high, trend_low, deleted_high, deleted_low = [], [], [], []

    last_filtered_high, last_filtered_low = filter_trend_initial(
        trend_high, trend_low, data, trend_config
    )

    return trend_high, trend_low, last_filtered_high, last_filtered_low
