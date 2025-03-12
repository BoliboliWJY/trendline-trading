import os
import time
import numpy as np
from src.get_data.backtest_filename_getter import get_backtest_filename
from src.get_data.get_data_in_batches import get_data_in_batches

def data_getter(client,coin_type,interval,length,backtest_start_time,backtest_end_time,contract_type):
    filename, typename = get_backtest_filename(coin_type, interval, length, backtest_start_time, backtest_end_time,contract_type)
    
    if os.path.exists(filename) and os.path.exists(typename):
        data = np.load(filename)
        type_data = np.load(typename)
    else:
        limit = length if length < 1000 else 1000
        data, type_data = get_data_in_batches(client,coin_type,interval,length,backtest_start_time,limit)
        np.save(filename, data)
        np.save(typename, type_data)
        
    return data, type_data
