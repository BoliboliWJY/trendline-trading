import os

def get_backtest_filename(coin_type, interval, length, backtest_start_time, backtest_end_time):
    backtest_dir = os.path.join(os.getcwd(), 'backtest\\k_lines')
    os.makedirs(backtest_dir, exist_ok=True)
    directory = os.path.join(backtest_dir, coin_type)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{coin_type}_{interval}_{length}_{backtest_start_time}_{backtest_end_time}.npy")
    typename = os.path.join(directory, f"{coin_type}_{interval}_{length}_{backtest_start_time}_{backtest_end_time}_type.npy")
    return filename, typename
    