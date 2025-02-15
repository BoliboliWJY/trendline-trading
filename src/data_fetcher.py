import numpy as np
from .time_number import time_number

def get_data_in_batches(client,coin_type,interval,total_length,current_time,limit):
    """获取行情数据

    Args:
        client (Spot): 库内容
        coin_type (string): 币种类型
        interval (string): 间隔
        total_length (int): 请求总数据大小
        current_time (int): 当前时间，用于向前推导出起始时间
        limit (int): 步长，单次请求跨度

    Returns:
        arrary: 获取结果数据
    """
    # time_choice = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    # multiple_factor = [100, 300, 500, 1500, 3000, 6000, 12000, 24000, 36000, 48000, 72000, 720*400]#每类存活时长
    # initial_index = time_choice.index(interval)
    
    total_batches = total_length // limit
    left_num = total_length % limit
    data = np.zeros((total_length, 12))
    for i in range(total_batches):
        start_time = current_time - time_number(interval) * (limit * (i + 1) + left_num)
        end_time = start_time + time_number(interval) * limit
        # print(start_time)
        batch_data = np.array(client.klines(symbol=coin_type, interval=interval, startTime=start_time*1000, endTime=end_time * 1000, limit=limit))
        start_idx = (total_batches - 1 - i) * limit
        end_idx = start_idx + batch_data.shape[0]
        data[start_idx:end_idx, :] = batch_data
    
    if left_num != 0:
        start_time = current_time - time_number(interval) * left_num
        end_time = start_time + time_number(interval) * left_num
        batch_data = np.array(client.klines(symbol=coin_type, interval=interval,    startTime=start_time*1000, endTime=end_time * 1000,  limit=left_num))
        start_idx = total_length - left_num
        end_idx = start_idx + left_num
        data[start_idx:end_idx, :] = batch_data
    

    type_data = np.zeros((total_length, 1))
    for i in range(1, total_length):
        if data[i, 1] <= data[i, 4]:
            type_data[i] = 1
      
    # 原始数据排列顺序为：[0]开盘时间、[1]开盘价、[2]最高价、[3]最低价、[4]收盘价(当前K线未结束的即为最新价)、[5]成交量、[6]收盘时间、[7]成交额、[8]成交笔数、[9]主动买入成交量、[10]主动买入成交额、[11]请忽略该参数
    data[:,0] = (data[:,0] + data[:,6]) / 2 #取时间中值,减小误差
    # data = data[:, [0, 2, 3, 1, 4, 6]].astype(float)#Ave_ime,High,Low,Open,Close,Close_time
    data = data[:, [0, 2, 3, 1, 4, 5, 7, 8, 9, 10]].astype(float)# 0:时间,1:最高价,2:最低价,3:开盘价,4:收盘价,5:成交量,6:成交额,7:成交笔数,8:主动买入成交量,9:主动买入成交额
    # if(interval == '1s'):
    # data = data_compression(data)
    # data = data[:,[0,1]]
    return data, type_data.reshape(-1)
