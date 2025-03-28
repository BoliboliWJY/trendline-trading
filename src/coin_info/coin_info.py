import math
class CoinInfo:
    def __init__(self, client, coin_type, min_amount, use_max_leverage=False):
        """初始化币种信息

        Args:
            client (UMFutures): 币安客户端
            coin_type (str): 币种名称
            min_amount (float): 可用金额
            use_max_leverage (bool): 是否使用最大杠杆
        """
        self.client = client
        self.coin_type = coin_type
        self.min_amount = min_amount
        
        exchange_info = client.exchange_info()
        target_info = None
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["symbol"] == coin_type:
                target_info = symbol_info
                break
        
        self.min_qty = None
        self.step_size = None
        self.price_precision = None
        for flit in target_info["filters"]:
            if flit["filterType"] == "LOT_SIZE":
                self.min_qty = float(flit["minQty"])
                self.step_size = float(flit["stepSize"])
            elif flit["filterType"] == "PRICE_FILTER":
                self.price_precision = float(flit["tickSize"])
            
        leverage_info = client.leverage_brackets(symbol=coin_type)
        self.applicable_leverage = None
        
        if use_max_leverage:
            # 使用最大可用杠杆
            max_leverage = 1
            for bracket in leverage_info[0]["brackets"]:
                if min_amount * bracket["initialLeverage"] <= float(bracket["notionalCap"]):
                    max_leverage = bracket["initialLeverage"]
                    break
            self.applicable_leverage = max_leverage
        else:
            # 原来的逻辑：根据总金额选择杠杆
            for bracket in leverage_info[0]["brackets"]:
                if min_amount <= float(bracket["notionalCap"]):
                    self.applicable_leverage = bracket["initialLeverage"]
                    break
            if self.applicable_leverage is None:
                self.applicable_leverage = 1
        
    def get_min_qty(self):
        current_price = self.client.ticker_price(symbol = self.coin_type)["price"]
        
        # 计算实际可用的总金额（考虑杠杆）
        total_available = self.min_amount * self.applicable_leverage
        # 使用实际可用总金额计算数量
        raw_qty = max((total_available / float(current_price)), self.min_qty)
        precision = int(round(-math.log10(self.step_size), 0))
        adjusted_qty = math.ceil(raw_qty * (10 ** precision)) / (10 ** precision)
        
        return adjusted_qty
        
    def get_price_precision(self):
        """获取价格精度

        Returns:
            float: 价格精度（tickSize）
        """
        return self.price_precision
        
    

