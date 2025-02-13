import time
import numpy as np
from src.data_fetcher import get_data_in_batches

def get_latest_klines(client, coin_type, interval, total_length):
    """获取最新K线数据
    Args:
        client:库
        coin_type:目标币种
        interval:时间间隔
        total_length:总长度
    Returns:
        data:价格数据
        type_data:升/降价格数据
        """
    current_time = int(time.time())
    limit = total_length if total_length < 1000 else 1000
    data, type_data = get_data_in_batches(client, coin_type, interval, total_length, current_time, limit)
    return data, type_data
