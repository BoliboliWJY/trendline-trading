def get_current_price(client, coin_type):
    """获取最新价格数据
    Args:
        client:库
        coin_type:目标币种
    Returns:
        current_price:最新价格
    """
    current_price_info = client.get_symbol_ticker(symbol=coin_type)
    current_price = current_price_info['price']
    return current_price

# print(get_current_price(client, coin_type))