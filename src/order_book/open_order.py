# 对开仓信号进行处理，生成订单簿

def open_order(open_signal, client, coin_type, contract_type, coin_info, take_profit_percent=0.005, stop_loss_percent=0.002):
    if open_signal["high_open"]:
        order_type = "SELL"
        position_side = "SHORT"
    elif open_signal["low_open"]:
        order_type = "BUY"
        position_side = "LONG"
    else:
        # raise ValueError(f"Invalid open signal: {open_signal}")
        return None

    min_qty = coin_info.get_min_qty()
    
    # 获取当前价格
    current_price = float(client.ticker_price(symbol=coin_type)["price"])
    
    # 获取价格精度
    price_precision = coin_info.get_price_precision()
    if price_precision < 1:
        import math
        price_precision_digits = abs(int(math.log10 (price_precision)))
    else:
        price_precision_digits = int(price_precision)
    
    # 计算止盈止损价格
    if position_side == "LONG":
        take_profit_price = round(current_price * (1 + take_profit_percent), price_precision_digits)
        stop_loss_price = round(current_price * (1 - stop_loss_percent), price_precision_digits)
    else:  # SHORT
        take_profit_price = round(current_price * (1 - take_profit_percent), price_precision_digits)
        stop_loss_price = round(current_price * (1 + stop_loss_percent), price_precision_digits)
    
    try:
        # 执行市价下单
        order_response = client.new_order(
            symbol=coin_type,
            side=order_type,
            positionSide=position_side,
            type="MARKET",
            quantity=min_qty,
            leverage=coin_info.applicable_leverage
        )
        
        # 获取成交价格
        entry_price = float(order_response["avgPrice"]) if "avgPrice" in order_response else current_price
        
        # 设置止盈订单
        take_profit_order = client.new_order(
            symbol=coin_type,
            side="BUY" if order_type == "SELL" else "SELL",  # 与开仓方向相反
            positionSide=position_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=take_profit_price,
            closePosition=True,  # 平掉整个仓位
            timeInForce="GTE_GTC",  # 订单会一直有效，直到被成交或取消
            workingType="MARK_PRICE"  # 使用标记价格触发
        )
        
        # 设置止损订单
        stop_loss_order = client.new_order(
            symbol=coin_type,
            side="BUY" if order_type == "SELL" else "SELL",  # 与开仓方向相反
            positionSide=position_side,
            type="STOP_MARKET",
            stopPrice=stop_loss_price,
            closePosition=True,  # 平掉整个仓位
            timeInForce="GTE_GTC",  # 订单会一直有效，直到被成交或取消
            workingType="MARK_PRICE"  # 使用标记价格触发
        )
        
        # 将止盈止损订单信息添加到开仓订单响应中
        order_response["take_profit_order"] = take_profit_order
        order_response["stop_loss_order"] = stop_loss_order
        order_response["take_profit_price"] = take_profit_price
        order_response["stop_loss_price"] = stop_loss_price
        
        return order_response
        
    except Exception as e:
        print(f"下单或设置止盈止损失败: {e}")
        return None
