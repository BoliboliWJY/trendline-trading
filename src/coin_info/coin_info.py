import math
class CoinInfo:
    def __init__(self, client, coin_type, contract_type, min_amount):
        """初始化币种信息

        Args:
            client (UMFutures): 币安客户端
            coin_type (str): 币种名称
            contract_type (str): 合约类型
            min_amount (float): 最小交易金额
        """
        self.client = client
        self.coin_type = coin_type
        self.contract_type = contract_type
        self.min_amount = min_amount
        
        exchange_info = client.exchange_info()
        target_info = None
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == coin_type:
                target_info = symbol_info
                break
        
        self.min_qty = None
        self.step_size = None
        for flit in target_info["filters"]:
            if flit["filterType"] == "LOT_SIZE":
                self.min_qty = float(flit["minQty"])
                self.step_size = float(flit["stepSize"])
                break
            
        leverage_info = client.leverage_brackets(symbol=coin_type)
        self.applicable_leverage = None
        for bracket in leverage_info[0]["brackets"]:
            if min_amount <= float(bracket["notionalCap"]):
                self.applicable_leverage = bracket["initialLeverage"]
                break
        if self.applicable_leverage is None:
            self.applicable_leverage = 1
        
    def get_min_qty(self):
        current_price = self.client.ticker_price(symbol = self.coin_type)["price"]
        
        raw_qty = max((100 / float(current_price)), self.min_qty)
        precision = int(round(-math.log10(self.step_size), 0))
        adjusted_qty = math.ceil(raw_qty * (10 ** precision)) / (10 ** precision)
        
        return adjusted_qty
        
    

