from src.filter.filters import filter_trend_initial

def compute_initial_trends(current_data, trend_generator, data, trend_config, last_filtered_high, last_filtered_low):
    """
    Computes the initial trends using the trend generator.
    This method can also be used for live trading initialization.
    Args:
        current_data: the current data.
        trend_generator: the trend generator.
        data: the data.
        trend_config: the trend config.
        last_filtered_high: the last filtered high trend data.
        last_filtered_low: the last filtered low trend data.
    Returns:
        trend_high, trend_low: the computed trend data after initial filtering.
        last_filtered_high, last_filtered_low: the last filtered trend data.
    """
    try:
        for _ in range(len(current_data) - 1):

            trend_high, trend_low, deleted_high, deleted_low = next(
                trend_generator
            )
    except StopIteration:
        trend_high, trend_low, deleted_high, deleted_low = [], [], [], []

    # last_filtered_high, last_filtered_low = filter_trend_initial(
    #     trend_high, trend_low, data, trend_config
    # )
    trend_high, trend_low = filter_trend_initial(
        trend_high, trend_low, data, trend_config
    )
    last_filtered_high = trend_high
    last_filtered_low = trend_low

    return trend_high, trend_low, last_filtered_high, last_filtered_low
