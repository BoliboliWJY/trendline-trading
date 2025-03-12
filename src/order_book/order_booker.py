class OrderBook:
    """
    用于币安币本位交易的订单簿类

    属性：
        order_id (str/int): 订单编号
        order_type (str): 订单类型 ("buy" 或 "sell")
        open_price (float): 开仓价格
        quantity (float): 订单数量
        open_time (str): 开仓时间
        status (str): 订单状态，初始化为 "open"
        stop_loss (float): 止损价格
        take_profit (float): 止盈价格
        close_price (float): 平仓价格
        close_time (str): 平仓时间
        realized_profit (float): 实际盈亏
        trade_logs (list): 记录订单操作和交易记录的日志列表
        extra_info (dict): 扩展的空信息，用于后续扩展其他数据
    """

    def __init__(self, order_id, order_type, open_price, quantity, open_time,
                 stop_loss=None, take_profit=None):
        self.order_id = order_id
        self.order_type = order_type  # "buy" 或 "sell"
        self.open_price = open_price
        self.quantity = quantity
        self.open_time = open_time
        self.status = "open"  # 订单刚创建时状态为 open
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 初始化附加的空信息字典（后续可以存储更多扩展信息）
        self.extra_info = {}

        # 初始化交易日志为空列表
        self.trade_logs = []

        # 平仓信息及实际盈亏初始化为空
        self.close_price = None
        self.close_time = None
        self.realized_profit = None

        # 记录订单初始化日志
        self._log_trade(
            action="初始化订单",
            price=open_price,
            time=open_time,
            quantity=quantity,
            fee=0,
            note="订单创建，初始化扩展信息为空"
        )

    def _log_trade(self, action, price, time, quantity, fee=0, note=None):
        """
        内部方法：记录每一次订单操作的详细信息

        参数:
            action (str): 操作动作，如 "初始化订单"、"更新止损"、"部分成交"、"平仓订单" 等
            price (float): 操作涉及的价格
            time (str): 操作时间
            quantity (float): 操作数量或影响数量
            fee (float): 相关手续费（默认为0）
            note (str): 附加备注信息
        """
        log_entry = {
            "action": action,
            "price": price,
            "time": time,
            "quantity": quantity,
            "fee": fee,
            "note": note
        }
        self.trade_logs.append(log_entry)

    def update_stop_loss(self, new_stop_loss, update_time):
        """
        更新订单的止损价格

        参数：
            new_stop_loss (float): 新的止损价格
            update_time (str): 更新时间
        """
        old_stop_loss = self.stop_loss
        self.stop_loss = new_stop_loss
        self._log_trade(
            action="更新止损",
            price=new_stop_loss,
            time=update_time,
            quantity=self.quantity,
            fee=0,
            note=f"旧止损: {old_stop_loss}"
        )

    def update_take_profit(self, new_take_profit, update_time):
        """
        更新订单的止盈价格

        参数：
            new_take_profit (float): 新的止盈价格
            update_time (str): 更新时间
        """
        old_take_profit = self.take_profit
        self.take_profit = new_take_profit
        self._log_trade(
            action="更新止盈",
            price=new_take_profit,
            time=update_time,
            quantity=self.quantity,
            fee=0,
            note=f"旧止盈: {old_take_profit}"
        )

    def record_partial_fill(self, fill_price, fill_quantity, fill_time, fee=0):
        """
        记录部分成交或追加头寸的操作记录

        参数：
            fill_price (float): 成交均价
            fill_quantity (float): 成交数量
            fill_time (str): 成交时间
            fee (float): 手续费（默认为0）
        """
        self._log_trade(
            action="部分成交",
            price=fill_price,
            time=fill_time,
            quantity=fill_quantity,
            fee=fee,
            note="部分成交/加仓"
        )

    def close_order(self, close_price, close_time, fee=0):
        """
        平仓订单，记录平仓细节并计算实际盈亏

        参数：
            close_price (float): 平仓价格
            close_time (str): 平仓时间
            fee (float): 平仓手续费（默认为0）
        """
        if self.status != "open":
            raise Exception("订单状态异常：订单已平仓或操作已结束。")

        self.close_price = close_price
        self.close_time = close_time
        self.status = "closed"

        # 根据订单类型计算盈亏
        if self.order_type == "buy":
            profit = (close_price - self.open_price) * self.quantity
        elif self.order_type == "sell":
            profit = (self.open_price - close_price) * self.quantity
        else:
            profit = 0

        # 考虑手续费后的盈亏计算
        profit -= fee
        self.realized_profit = profit

        self._log_trade(
            action="平仓订单",
            price=close_price,
            time=close_time,
            quantity=self.quantity,
            fee=fee,
            note=f"实际盈亏: {profit}"
        )

    def get_order_details(self):
        """
        获取订单的所有详细信息及交易日志

        返回：
            dict：包含订单所有属性及交易记录信息
        """
        return {
            "order_id": self.order_id,
            "order_type": self.order_type,
            "open_price": self.open_price,
            "quantity": self.quantity,
            "open_time": self.open_time,
            "status": self.status,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "close_price": self.close_price,
            "close_time": self.close_time,
            "realized_profit": self.realized_profit,
            "trade_logs": self.trade_logs,
            "extra_info": self.extra_info
        }

    def __repr__(self):
        return (
            f"Order(id={self.order_id}, type={self.order_type}, open_price={self.open_price}, "
            f"quantity={self.quantity}, open_time={self.open_time}, status={self.status}, "
            f"stop_loss={self.stop_loss}, take_profit={self.take_profit}, close_price={self.close_price}, "
            f"close_time={self.close_time}, realized_profit={self.realized_profit})"
        )


if __name__ == "__main__":
    # 示例：创建并操作一个币安币本位交易订单

    # 创建一个买入订单
    order = OrderBook(
        order_id=1002,
        order_type="buy",
        open_price=50000,
        quantity=0.1,
        open_time="2023-10-15 11:00:00",
        stop_loss=49000,
        take_profit=52000
    )
    print(order)

    # 更新止损价格
    order.update_stop_loss(new_stop_loss=49500, update_time="2023-10-15 11:05:00")

    # 更新止盈价格
    order.update_take_profit(new_take_profit=52500, update_time="2023-10-15 11:10:00")

    # 记录部分成交/追加头寸
    order.record_partial_fill(fill_price=50100, fill_quantity=0.05, fill_time="2023-10-15 11:15:00", fee=1)

    # 平仓订单
    order.close_order(close_price=51500, close_time="2023-10-15 11:30:00", fee=2)

    # 输出订单详细信息及交易日志
    details = order.get_order_details()
    print(details)
     
     
        
