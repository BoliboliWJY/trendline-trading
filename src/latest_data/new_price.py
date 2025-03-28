
class NewPrice():
    def __init__(self, client, coin_type, contract_type):
        self.client = client
        self.coin_type = coin_type
        self.contract_type = contract_type
        
        ping_result = self.client.ping()
        print(ping_result)

    def __next__(self):
        """获取最新价格数据
        Args:
            client:库
        coin_type:目标币种
    Returns:
        current_price:最新价格
        """
        if self.contract_type == 'um':
            # current_price_info = self.client.get_symbol_ticker(symbol=self.coin_type)
            current_price_info = self.client.ticker_price(symbol=self.coin_type)
        else:
            current_price_info = self.client.ticker_price(symbol=self.coin_type)
        current_price = current_price_info
        return current_price
        
