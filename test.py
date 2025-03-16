# import requests
# print("真实出口IP:", requests.get('https://api.ipify.org').text)

import datetime
import time
from binance.um_futures import UMFutures
import math
# api_key = "OqQibH7VeZp0zaWix1Xc4oFKdtpnp9KE8pzHiYIZuQN1za6Ip0VJxKlbwTyfefis"
# api_secret = "lytyKr5kvRbNepKJOIAONUkC9yJ819HcBIrlhQwiZ07wicz3jqXZfaIvqG5mE3gd"
# client = UMFutures(api_key, api_secret)
api_key = "2e54bffe0f994be734967a4ea6786b50954bc0229d815d64fb067511b8733be1"
api_secret = "71b59ea564cb76df48ba5d83090e585cdd0b3b02f1923034169f90a48279cd91"
client = UMFutures(api_key, api_secret, base_url="https://testnet.binancefuture.com")
# 使用 base_url 指定测试网地址

print(client.account()["totalMarginBalance"])

target_symbol = "BTCUSDT"

# 切换仓位模式为逐仓模式（ISOLATED）
try:
    response = client.change_margin_type(symbol=target_symbol, marginType="ISOLATED")
    print("成功切换至逐仓模式:", response)
except Exception as e:
    print("切换仓位模式失败（可能已经是逐仓模式）：", e)

# 调整杠杆至125倍
try:
    response = client.change_leverage(symbol=target_symbol, leverage=125)
    print("杠杆调整成功:", response)
except Exception as e:
    print("杠杆调整失败:", e)

exchange_info = client.exchange_info()
target_info = None
for symbol_info in exchange_info["symbols"]:
    if symbol_info["symbol"] == target_symbol:
        target_info = symbol_info
        break

min_qty = None
step_size = None
for flit in target_info["filters"]:
    if flit["filterType"] == "LOT_SIZE":
        min_qty = float(flit["minQty"])
        step_size = float(flit["stepSize"])
        break

print(f"{target_symbol} 的最小交易量: {min_qty}")

leverage_info = client.leverage_brackets(symbol=target_symbol)
# print(leverage_info)
max_leverage = leverage_info[0]["brackets"][0]["initialLeverage"]
max_qty = leverage_info[0]["brackets"][0]["notionalCap"]
print(f"最高杠杆: {max_leverage}")
print(f"最大数量: {max_qty}")

current_price = client.ticker_price(symbol=target_symbol)["price"]
print(f"当前价格: {current_price}")

# 计算所需数量，保证下单金额不少于 100 美元，并且不低于最小交易量要求
raw_qty = max((100 / float(current_price)), min_qty)

# 根据 step_size 确定允许的小数位数
precision = int(round(-math.log10(step_size), 0))

# 使用 math.ceil 向上取整，以确保下单的金额不少于 100 美元
adjusted_qty = math.ceil(raw_qty * (10 ** precision)) / (10 ** precision)

print(f"调整后的最小订单数量: {adjusted_qty}")

order_response = client.new_order(
    symbol=target_symbol,
    side="BUY",
    type="MARKET",
    quantity=adjusted_qty,
    leverage=max_leverage
)
print(order_response)

# exchange_info = client.exchange_info()
# for symbol_info in exchange_info["symbols"]:
#     if symbol_info["symbol"] == "BTCUSDT":
#         print(symbol_info)

# print(client.balance())

# account =client.account()
# print(account["totalMarginBalance"])

# print(client.leverage_brackets(symbol="BTCUSDT"))

# if btc_info is None:
#     print("未找到 BTCUSDT 交易对")
#     exit()

# # 提取交易规则（这里取 pricePrecision 和 quantityPrecision）
# price_precision = btc_info.get("pricePrecision", 2)      # 默认值2
# quantity_precision = btc_info.get("quantityPrecision", 3)  # 默认值3

# print("BTCUSDT 的价格精度：", price_precision)
# print("BTCUSDT 的数量精度：", quantity_precision)

# # 定义订单参数（需要满足交易规则）
# raw_quantity = 0.01   # 示例数量，至少0.001
# raw_stop_price = 10000  # 示例止损价格

# # 按照交易对要求的精度对参数进行截断
# quantity = math.floor(raw_quantity * (10 ** quantity_precision)) / (10 ** quantity_precision)
# stop_price = math.floor(raw_stop_price * (10 ** price_precision)) / (10 ** price_precision)

# print("调整后的订单数量：", quantity)
# print("调整后的止损价格：", stop_price)

# # 发起模拟下单（跟踪止损市价单）
# try:
#     response = client.new_order(
#         symbol="BTCUSDT",
#         leverage=10,
#         side="BUY",
#         type="",
#         quantity=quantity,
#         stopPrice=stop_price,
#         callbackRate=0.2  # 通常也要满足精度要求
#     )
#     print("模拟下单返回：", response)
# except Exception as e:
#     print("模拟下单出错：", e)
# # 使用高速轮询获取 BTC 价格和服务器时间
# while True:
#     try:
#         # 获取 BTC 价格，使用正确的方法 ticker_price
#         btc_price = client.ticker_price(symbol='BTCUSDT')
    
#         # 获取 Binance 服务器时间（单位为毫秒），使用正确的方法 time
#         # server_time_dict = client.time()
#         # server_time_ms = server_time_dict["serverTime"]
#         # # 将毫秒转换为秒并转换为 datetime 对象
#         # server_time = datetime.datetime.fromtimestamp(server_time_ms / 1000.0)
#         # current_server_time = server_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
    
#         # print("BTC 价格：", btc_price, "服务器时间（毫秒）：", current_server_time)
#         current_time = datetime.datetime.now()
#         print("BTC 价格：", btc_price, "当前时间：", current_time)
    
#         # 设置较短延时，调整轮询频率（此处为 100ms，可以根据需要调整）
#         time.sleep(0.1)
        
#     except Exception as e:
#         print("请求异常:", e)
#         # 请求异常时稍作延时再继续轮询
#         time.sleep(1)


# # btc_book_ticker = client.book_ticker(symbol='BTCUSDT')
# # print(btc_book_ticker)
# # btc_price= client.time()
# # print(btc_price)

# test_order = client.get_all_orders(symbol='BTCUSDT')
# print(test_order)
