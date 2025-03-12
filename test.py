# import requests
# print("真实出口IP:", requests.get('https://api.ipify.org').text)

import datetime
import time
from binance.um_futures import UMFutures

api_key = "Y8l83CSLEGR6C4ZQym5NANoZ0x7bTbcBJkHfxW7G85OxCktCar9IPUdIbINb9K9e"
api_secret = "GmkTBlabiNvq75nQBhRdiR91G86DZSKYogL8jXikFlWmdvRTOLu3XQNTXqTZdltg"

client = UMFutures(api_key, api_secret)

# 使用高速轮询获取 BTC 价格和服务器时间
while True:
    try:
        # 获取 BTC 价格，使用正确的方法 ticker_price
        btc_price = client.ticker_price(symbol='BTCUSDT')
    
        # 获取 Binance 服务器时间（单位为毫秒），使用正确的方法 time
        # server_time_dict = client.time()
        # server_time_ms = server_time_dict["serverTime"]
        # # 将毫秒转换为秒并转换为 datetime 对象
        # server_time = datetime.datetime.fromtimestamp(server_time_ms / 1000.0)
        # current_server_time = server_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
    
        # print("BTC 价格：", btc_price, "服务器时间（毫秒）：", current_server_time)
        current_time = datetime.datetime.now()
        print("BTC 价格：", btc_price, "当前时间：", current_time)
    
        # 设置较短延时，调整轮询频率（此处为 100ms，可以根据需要调整）
        time.sleep(0.1)
    except Exception as e:
        print("请求异常:", e)
        # 请求异常时稍作延时再继续轮询
        time.sleep(1)


# btc_book_ticker = client.book_ticker(symbol='BTCUSDT')
# print(btc_book_ticker)
# btc_price= client.time()
# print(btc_price)

test_order = client.get_all_orders(symbol='BTCUSDT')
print(test_order)
