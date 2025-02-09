import os
import time
import numpy as np
from binance import Client
from src.data_fetcher import get_data_in_batches
from src.config_manager import get_backtest_filename

def load_or_fetch_data(client, coin_type, interval, total_length):
    """
    检查数据文件是否存在。如果存在则加载数据，否则调用 Binance 接口获取数据。
    """
    current_time = int(time.time())
    filename, typename = get_backtest_filename(coin_type, interval, total_length)
    
    if os.path.exists(filename) and os.path.exists(typename):
        # 如果存在，则加载数据
        data = np.load(filename)
        type_data = np.load(typename)
    else:
        # 如果数据文件不存在，则获取数据
        limit = total_length if total_length < 1000 else 1000
        data, type_data = get_data_in_batches(client, coin_type, interval, total_length, current_time, limit)
        np.save(filename, data)
        np.save(typename, type_data)
    
    return data, type_data