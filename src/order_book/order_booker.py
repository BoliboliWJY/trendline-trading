
class Order:
    def __init__(self, order_id, order_type, price, tick_time, stop_loss=None, drawdown=None, profit=None):
        self.order_id = order_id      # 订单 id
        self.order_type = order_type  # 'buy' 或 'sell'
        self.price = price            # 开仓价格
        self.tick_time = tick_time    # 开仓对应的 tick 时间
        self.status = "open"
        self.stop_loss = stop_loss
        self.max_drawdown = drawdown
        self.max_profit = profit
        
    def __repr__(self):
        return (f"Order(id={self.order_id}, type={self.order_type}, price={self.price}, "
                f"time={self.tick_time}, status={self.status}, stop_loss={self.stop_loss}, "
                f"drawdown={self.drawdown}, profit={self.profit})")
     
     
        
