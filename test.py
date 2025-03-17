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

# 设置止盈止损百分比
take_profit_percent = 0.005  
stop_loss_percent = 0.002    

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
price_precision = None

for flit in target_info["filters"]:
    if flit["filterType"] == "LOT_SIZE":
        min_qty = float(flit["minQty"])
        step_size = float(flit["stepSize"])
    elif flit["filterType"] == "PRICE_FILTER":
        price_precision = float(flit["tickSize"])

print(f"{target_symbol} 的最小交易量: {min_qty}")
print(f"{target_symbol} 的价格精度: {price_precision}")

leverage_info = client.leverage_brackets(symbol=target_symbol)
# print(leverage_info)
max_leverage = leverage_info[0]["brackets"][0]["initialLeverage"]
max_qty = leverage_info[0]["brackets"][0]["notionalCap"]
print(f"最高杠杆: {max_leverage}")
print(f"最大数量: {max_qty}")

current_price = float(client.ticker_price(symbol=target_symbol)["price"])
print(f"当前价格: {current_price}")

# 计算所需数量，保证下单金额不少于 100 美元，并且不低于最小交易量要求
raw_qty = max((100 / current_price), min_qty)

# 根据 step_size 确定允许的小数位数
qty_precision = int(round(-math.log10(step_size), 0))
price_precision_digits = int(round(-math.log10(price_precision), 0))

# 使用 math.ceil 向上取整，以确保下单的金额不少于 100 美元
adjusted_qty = math.ceil(raw_qty * (10 ** qty_precision)) / (10 ** qty_precision)

print(f"调整后的最小订单数量: {adjusted_qty}")

# 计算止盈止损价格
side = "BUY"  # 买入做多
position_side = "LONG"

if side == "BUY" and position_side == "LONG":
    take_profit_price = round(current_price * (1 + take_profit_percent), price_precision_digits)
    stop_loss_price = round(current_price * (1 - stop_loss_percent), price_precision_digits)
elif side == "SELL" and position_side == "SHORT":
    take_profit_price = round(current_price * (1 - take_profit_percent), price_precision_digits)
    stop_loss_price = round(current_price * (1 + stop_loss_percent), price_precision_digits)

print(f"开仓价格: {current_price}")
print(f"止盈价格 ({take_profit_percent*100}%): {take_profit_price}")
print(f"止损价格 ({stop_loss_percent*100}%): {stop_loss_price}")

# 执行市价下单
try:
    order_response = client.new_order(
        symbol=target_symbol,
        side=side,
        positionSide=position_side,
        type="MARKET",
        quantity=adjusted_qty,
        leverage=max_leverage
    )
    print("市价下单成功:", order_response)
    
    # 获取成交价格
    entry_price = float(order_response["avgPrice"]) if "avgPrice" in order_response else current_price
    
    # 设置止盈订单
    take_profit_order = client.new_order(
        symbol=target_symbol,
        side="SELL" if side == "BUY" else "BUY",  # 与开仓方向相反
        positionSide=position_side,
        type="TAKE_PROFIT_MARKET",
        stopPrice=take_profit_price,
        closePosition=True,  # 平掉整个仓位
        timeInForce="GTE_GTC",  # 订单会一直有效，直到被成交或取消
        workingType="MARK_PRICE"  # 使用标记价格触发
    )
    print("止盈订单设置成功:", take_profit_order)
    
    # 设置止损订单
    stop_loss_order = client.new_order(
        symbol=target_symbol,
        side="SELL" if side == "BUY" else "BUY",  # 与开仓方向相反
        positionSide=position_side,
        type="STOP_MARKET",
        stopPrice=stop_loss_price,
        closePosition=True,  # 平掉整个仓位
        timeInForce="GTE_GTC",  # 订单会一直有效，直到被成交或取消
        workingType="MARK_PRICE"  # 使用标记价格触发
    )
    print("止损订单设置成功:", stop_loss_order)
    
except Exception as e:
    print("下单或设置止盈止损失败:", e)
