from src.binance_client import create_client
from src.data_fetcher import get_data_in_batches
from src.plotter import plot_data
from config.config import API_KEY, API_SECRET
from .utils import time_number
from src.trend_process import realtime_trend