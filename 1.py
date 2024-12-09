# %%
from binance.spot import Spot as Client
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
#%%
from math import ceil,floor
import numpy as np

def time_number(interval):
    """ 将时间间隔单位转换为秒数 """
    try:
        time_unit = interval[-1]
        time_value = int(interval[:-1])
    except (ValueError, IndexError):
        raise ValueError("输入格式不正确，应为类似于 '5m', '1h' 等格式")

    # 根据时间单位确定每步的秒数
    if time_unit == 's':  # 秒
        step_seconds = time_value
    elif time_unit == 'm':  # 分钟
        step_seconds = time_value * 60
    elif time_unit == 'h':  # 小时
        step_seconds = time_value * 3600
    elif time_unit == 'd':  # 天
        step_seconds = time_value * 86400
    elif time_unit == 'w':  # 周
        step_seconds = time_value * 604800
    else:
        raise ValueError("不支持的时间单位")
    return step_seconds
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
    data = data[:, [0, 2, 3, 1, 4]].astype(float)#Ave_ime,High,Low
    # if(interval == '1s'):
    # data = data_compression(data)
    # data = data[:,[0,1]]
    return data, type_data


#%%
#获取历史数据

key = 'uiY3WGKVNEaCkntmyikLCALO9O63PBAYcVDwLw0Xu66AgcrEBXab0UANMbWZOsj4'
secret = 'O7zn1HEFTr0e9msT1m52Nu6utZtIkmicRsbUtpSJSdVJrTlLs2NIVLLhiwALXKez'
client = Client(key, secret)
coin_type = "BTCUSDT"

mode = 'backtest' #'realtime' or 'backtest'
threshold = 0.0005
total_length = 1000
interval = '3m'
time_number(interval)
current_time = int(time.time())
limit = total_length if total_length < 1000 else 1000
# limit = 10
start_time = time.perf_counter()
[data, type_data] = get_data_in_batches(client,coin_type,interval,total_length,current_time,limit)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"获取数据耗时: {elapsed_time:.6f} 秒")
# plt.figure()
# plt.plot(data[:,0])
# plt.show()
# plt.plot(data[:,0],data[:,1])
# plt.plot(data[:,0],data[:,2])
# plt.show()
#%%
# data = np.array([
#     [-1,3,0],
#     [0,10,2],
#     [1,5,-2],
#     [2,3,1.5],
#     [3,6,0.5],
#     [4,3,2.5],
#     [5,6,4],
#     [6,7,5],
#     [7,4,3],
# ])
# threshold = 2


# plt.plot(data[:,0],data[:,2])
# plt.show()

from sortedcontainers import SortedList
#先处理最高价

def initial_slope(data, trend, idx):
    """初始化斜率数组

    Args:
        data (np.list): 数据，0为时间，1为最高价，2为最底下
        trend (sortedlist): 保存斜率数据
        idx (int): 选择第1或2列数据
    """
    for i in range(1, len(data)):
        j = i-1 
        current_time, current_value = data[i,[0,idx]]
        prev_time, prev_value = data[j,[0,idx]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        trend[i].add((slope,j))

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

def update_plot(trend_high, trend_low, i):
    ax.cla()
    ax.plot(data[:i+1, 0], data[:i+1, 1], marker='.', markersize=3, color='blue')
    ax.plot(data[:i+1, 0], data[:i+1, 2], marker='.', markersize=3, color='orange')
    for k in range(1, i+1):
        for slope, j in trend_high[k]:
            point = data[k, [0, idx_high]]
            m = slope
            ax.axline(point, slope=m, color='red', linewidth=0.1)
        for slope, j in trend_low[k]:
            point = data[k, [0, idx_low]]
            m = slope
            ax.axline(point, slope=m, color='green', linewidth=0.1)
            ax.set_title(f'Index i = {data[i,0]}')
            plt.draw()
         
def on_key(event):
    global i
    extended_num = 1
    if event.key == 'right':
        print(i)
        
        if i < len(data) - 1:
            i += 1
            if i > extended_num:
                new_trend(data[:i+1], update_trend_high, update_trend_low, initial_single_slope, trend_high[extended_num],  idx_high, trend_low[extended_num], idx_low)
                trend_high[i] = trend_high[extended_num]
                trend_low[i] = trend_low[extended_num]
            update_plot(trend_high=trend_high[i], trend_low=trend_low[i], i = i-1)
    elif event.key == 'left':
        print(i)
        if i > 0:
            i -= 1
            update_plot(trend_high=trend_high[i], trend_low=trend_low[i], i = i-1)
    elif event.key == 'escape':
        plt.close(fig)

def remove_elements_above_threshold(trend_high, threshold):
    for current_idx in range(len(trend_high) - 1, -1, -1):
        current_list = trend_high[current_idx]
        if any(abs(slope) >= threshold for slope, _ in current_list):
            del trend_high[current_idx]

def update_trend_high(data, trend_high, current_idx, i, current_slope, threshold):
    index = trend_high[i].bisect_left((current_slope,))
    for j in range(index - 1, -1, -1):
        current_time, current_value = data[current_idx, [0, 1]]
        prev_index = trend_high[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 1]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_high[i][j]
        # remove_elements_above_threshold(trend_high, threshold)
        trend_high[current_idx].add((slope, prev_index))
        update_trend_high(data, trend_high, current_idx, prev_index, slope, threshold)

def update_trend_low(data, trend_low, current_idx, i, current_slope, threshold):
    index = trend_low[i].bisect_right((current_slope,))
    if len(trend_low[i]) == 0:
        return
    for j in range(len(trend_low[i])-1, index-1, -1):
        current_time, current_value = data[current_idx, [0, 2]]
        prev_index = trend_low[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 2]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_low[i][j]
        # if abs(current_slope) >= threshold:
        #     del trend_low[current_idx][0]
        trend_low[current_idx].add((slope, prev_index))
        update_trend_low(data, trend_low, current_idx, prev_index, slope, threshold)

def new_trend(data, update_trend_high, update_trend_low, initial_single_slope, trend_high, idx_high, trend_low, idx_low):
    for i in range(1,len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        initial_single_slope(data[:i+1], trend_high, idx_high)
        current_slope = trend_high[i][0][0]
        update_trend_high(data,trend_high = trend_high,current_idx = i, i = i - 1, current_slope=current_slope)
    
        trend_low.append(SortedList(key=lambda x: x[0]))  
        initial_single_slope(data[:i+1], trend_low, idx_low)
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low = trend_low, current_idx = i, i = i - 1, current_slope=current_slope)

def price_visualize(data, type_data):
    """plot the price with high and low data in plt

    Args:
        data (np.array): market data
        type_data (np.array(bool)): rise or fall, based on (open - close)
    """
    x = data[:, 0]
    y1 = data[:, 1]
    y2 = data[:, 2]
    # 创建交替的 x 和 y 数据
    x_combined = np.empty((x.size + x.size,), dtype=x.dtype)
    y_combined = np.empty((y1.size + y2.size,), dtype=y1.dtype)
    x_combined[0::2] = x
    x_combined[1::2] = x
    y_combined[0::2] = y1
    y_combined[1::2] = y2
    for i in range(len(data)):
        if type_data[i]:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2], color='green')
        else:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2], color='red')
            
    x = data[:, 0]
    y1 = data[:, 3]
    y2 = data[:, 4]
    # 创建交替的 x 和 y 数据
    x_combined = np.empty((x.size + x.size,), dtype=x.dtype)
    y_combined = np.empty((y1.size + y2.size,), dtype=y1.dtype)
    x_combined[0::2] = x
    x_combined[1::2] = x
    y_combined[0::2] = y1
    y_combined[1::2] = y2
    for i in range(len(data)):
        if type_data[i]:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2],linewidth=3, color='green')
        else:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2],linewidth=3, color='red')

def trend_visualize(threshold, data, type_data, trend_high, trend_low):
    """visualize the trend data

    Args:
        threshold (int): the maxnium number of the slope
        data (array): market data
        type_data (array): rise or fall type
        trend_high (array): slope data for rise
        trend_low (array): slope data for fall
    """
    for i in range(1, len(data)-1):
        if type_data[i] != type_data[i - 1] or type_data[i] != type_data[i + 1]:
            for slope, j in trend_high[i]:
                if type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                    if abs(slope) > threshold or type_data[j] == 0:
                        continue
                    start_point = data[j, [0, 1]]
                    end_point = data[i, [0, 1]]
                    m = slope
                    plt.axline(end_point, slope=m, color='red', linewidth=0.1, label=f'Slope = {m}  ')
            for slope, j in trend_low[i]:
                if  type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                    if abs(slope) > threshold or type_data[j] == 1:
                        continue
                    point = data[i, [0, 2]]
                    m = slope
                    plt.axline(point, slope=m, color='green', linewidth=0.1, label=f'Slope = {m}')

# loop_times = []#计算耗时
#%%
# mean = sum(data[:, 1] - data[:, 2]) / len(data)
def calculate_trend(threshold, data, update_trend_high, update_trend_low, trend_high, trend_low, start_idx):
    """calculate rise/fall trend, it shall calculate the length of the data

    Args:
        threshold (double): maxinum number of slope
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
        update_trend_high(data, trend_high=trend_high, current_idx=i, i=i - 1, current_slope=current_slope, threshold=threshold)
    #process low_time
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low=trend_low, current_idx=i, i=i - 1, current_slope=current_slope, threshold=threshold)
    
    # end_time = time.perf_counter()  # 记录结束时间
    # elapsed_time = end_time - start_time  # 计算耗时
    # loop_times.append(elapsed_time)  # 将耗时添加到列表中
    # 可选：打印每次循环的耗时
    # print(f"循环索引 {i} 耗时: {elapsed_time:.6f} 秒")
    # print("calculating completed")
   
"""cumulative_times = np.cumsum(loop_times)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # 子图1：每次循环的耗时
    ax1.plot(loop_times, color='blue')
    ax1.grid(True)
    # 子图2：累计耗时
    ax2.plot(cumulative_times, color='green')
    ax2.grid(True)
    # 自动调整子图参数，防止标签重叠
    plt.tight_layout()
    # 显示图形
    plt.show()"""

#%%
mode = 'realtime'
mode = 'backtest'
if mode == 'realtime':
    # plt.ion()
    plt.figure()
    # plt.plot(data[:,0],data[:,1],marker = '.',markersize = 3)
    # plt.plot(data[:,0],data[:,2],marker = '.',markersize = 3)
    price_visualize(data, type_data)
    # plt.plot(x_combined, y_combined, marker='.', markersize=3)
    
    #initialize trend_high
    trend_high = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_high = 1
    initial_slope(data, trend_high, idx_high)
    #initialize trend_low
    trend_low = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_low = 2
    initial_slope(data, trend_low,idx_low)
    
    start_time = time.perf_counter()
    calculate_trend(threshold, data, update_trend_high, update_trend_low, trend_high, trend_low, 2)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"实测耗时: {elapsed_time:.6f} 秒")
    print("calculating completed")
    
    trend_visualize(threshold, data, type_data, trend_high, trend_low)
        # plt.draw()
        # plt.pause(1)  # 暂停以便观察更新，时间可以根据需要调整

    # plt.ioff()
    visual_number = total_length if total_length < 500 else 500 #单图内可视化数量
    plt.xlim(current_time * 1000 - visual_number * time_number(interval) * 1000, current_time * 1000 + time_number(interval) * 1000)
    plt.ylim(min(data[-visual_number:,2])* 0.9995, max(data[-visual_number:,1])* 1.0005)
    plt.show()
       
elif mode == 'backtest':
    idx_high = 1
    idx_low = 2
    trend_high = [[SortedList(key=lambda x: x[0])]]
    trend_low = [[SortedList(key=lambda x: x[0])]]
    
    start_time = time.perf_counter()
    for i in range(1,len(data)):
        trend_data = trend_high[i-1]
        trend_data.append(SortedList(key=lambda x: x[0]))
        trend_high.append(trend_data)
        
        trend_data = trend_low[i - 1]
        trend_data.append(SortedList(key=lambda x: x[0]))
        trend_low.append(trend_data)
        
        backtest_data = data[:i+1]
        initial_single_slope(backtest_data, trend=trend_high[i], idx=idx_high)
        initial_single_slope(backtest_data, trend=trend_low[i], idx=idx_low)
        if (i >= 2):
            calculate_trend(threshold=threshold, data=backtest_data, update_trend_high=update_trend_high, update_trend_low=update_trend_low, trend_high=trend_high[i], trend_low=trend_low[i],start_idx=i)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"回测耗时: {elapsed_time:.6f} 秒")
    print("calculating completed")
    
    #plot the trend data
    
    plt.figure()
    for i in range(floor(len(data) * 0.8), len(data)):
        plt.ion()
        plt.clf()
        backtest_data = data[:i+1]
        price_visualize(backtest_data, type_data)
        trend_visualize(threshold, backtest_data, type_data, trend_high[i], trend_low[i])
        visual_number = total_length if total_length < 100 else 100 #单图内可视化数量
        plt.xlim(backtest_data[-1][0] - visual_number * time_number(interval) * 1000, backtest_data[-1][0] +time_number(interval) * 1000)
        plt.ylim(min(backtest_data[-visual_number:,2])* 0.9995, max(backtest_data [-visual_number:,1])* 1.0005)
        plt.draw()
        plt.pause(0.1)  # 暂停以便观察更新，时间可以根据需要调整

    # Assume data is a NumPy array with columns: [time, high, low, close], etc.
    
        
        
        
    
    plt.show()


#%%