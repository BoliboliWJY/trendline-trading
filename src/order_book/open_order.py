# 对开仓信号进行处理，生成订单簿

def open_order(open_signal, client, coin_type, contract_type, coin_info):
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
    
    order_response = client.new_order(
        symbol=coin_type,
        side=order_type,
        positionSide=position_side,
        type="MARKET",
        quantity=min_qty,
        leverage=coin_info.applicable_leverage
    )
    
    return order_response
