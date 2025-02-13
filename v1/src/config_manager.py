import os
import yaml

def load_basic_config(config_path='config/basic_config.yaml'):
    """
    从配置文件中加载基本参数（包括API密钥、币种、目标时间、数据总长度、K线间隔等）
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def get_backtest_filename(coin_type, interval, total_length):
    """
    设置回测数据存储目录，返回数据文件和类型文件的路径
    """
    backtest_dir = os.path.join(os.getcwd(), 'backtest')
    os.makedirs(backtest_dir, exist_ok=True)
    directory = os.path.join(backtest_dir, coin_type)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{coin_type}_{interval}_{total_length}.npy")
    typename = os.path.join(directory, f"{coin_type}_{interval}_{total_length}_type.npy")
    return filename, typename