#%%
from sortedcontainers import SortedList
from src import create_client, API_KEY, API_SECRET, get_data_in_batches, plot_data, time_number, realtime_trend
import time
def main():
    client = create_client(API_KEY, API_SECRET)
    coin_type = "BTCUSDT"
    mode = 'realtime' #'realtime' or 'backtest'
    total_length = 200
    interval = '1m'
    time_number(interval)
    current_time = int(time.time())
    limit = total_length if total_length < 1000 else 1000
    # limit = 10
    data = get_data_in_batches(client,coin_type,interval,total_length,current_time,limit)
    if mode == 'realtime':
        realtime_trend(data)
    print(1)

if __name__ == "__main__":
    main()
    
    




