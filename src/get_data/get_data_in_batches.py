import numpy as np
from src.time_number import time_number


def get_data_in_batches(client, coin_type, interval, total_length, start_time, limit):
    """获取连续的K线数据，每次请求更新时间窗口

    Args:
        client (Spot): 客户端实例
        coin_type (str): 币种类型
        interval (str): 间隔字符串
        total_length (int): 请求数据总条数（总K线数）
        start_time (int): 起始时间戳（秒）
        limit (int): 单次请求的最大数量

    Returns:
        tuple: (data, type_data) 其中data为行情数据数组, type_data为附加的类型数据
    """
    # 预先分配空间
    data = np.zeros((total_length, 12))
    current_start_time = start_time
    num_fetched = 0

    # 循环获取整个区间的数据
    while num_fetched < total_length:
        batch_size = min(limit, total_length - num_fetched)
        batch_end_time = current_start_time + time_number(interval) * batch_size
        batch_data = np.array(
            client.klines(
                symbol=coin_type,
                interval=interval,
                startTime=current_start_time * 1000,
                endTime=batch_end_time * 1000,
                limit=batch_size,
            )
        )
        fetched = batch_data.shape[0]
        # 如果接口没有返回足够数据，提前退出
        if fetched == 0:
            break
        data[num_fetched : num_fetched + fetched, :] = batch_data
        num_fetched += fetched
        current_start_time = batch_end_time  # 更新起始时间

    # 计算附加的辅助数据，例如 type_data
    type_data = np.zeros((total_length, 1))
    for i in range(1, total_length):
        if data[i, 1] <= data[i, 4]:
            type_data[i] = 1

    # 整理数据排列顺序
    # 原始数据排列顺序为：[0]开盘时间、[1]开盘价、[2]最高价、[3]最低价、[4]收盘价、[5]成交量、[6]收盘时间、[7]成交额、[8]成交笔数、[9]主动买入成交量、[10]主动买入成交额、[11]请忽略该参数
    data[:, 0] = (data[:, 0] + data[:, 6]) / 2  # 取两个时间的中间值，减小误差
    data = data[:, [0, 2, 3, 1, 4, 5, 7, 8, 9, 10]].astype(float)
    return data, type_data.reshape(-1)
