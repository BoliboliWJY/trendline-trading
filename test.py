from binance.spot import Spot as Client
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%%
import numpy as np
import os
import yaml

from src.time_number import time_number
from src.data_fetcher import get_data_in_batches
#%%
#获取历史数据
total_result = [0, 0]
start_time = time.perf_counter()

with open('config/basic_config.yaml', 'r') as file:
    basic_config = yaml.safe_load(file)



key = basic_config['key']
secret = basic_config['secret']
coin_type = basic_config['coin_type']
aim_time = basic_config['aim_time']
total_length = basic_config['total_length']
interval = basic_config['interval']
current_time = int(time.time())
# limit = 10

client = Client(key, secret)

exchange_info = client.exchange_info()  # USDT本位合约
# 或者使用币本位合约：exchange_info = client.dapi_exchange_info()

# 提取所有的交易对名称
coin_types = [coin['symbol'] for coin in exchange_info['symbols']]
print(len(coin_types))