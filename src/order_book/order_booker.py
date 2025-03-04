
class Order:
    def __init__(self, order_id, order_type, price, tick_time):
        self.order_id = order_id      # 订单 id
        self.order_type = order_type  # 'buy' 或 'sell'
        self.price = price            # 开仓价格
        self.tick_time = tick_time    # 开仓对应的 tick 时间
        self.status = "open"
        
    def __repr__(self):
        return f"Order(id={self.order_id}, type={self.order_type}, price={self.price}, time={self.tick_time}, status={self.status})"

class OrderBook:
    def __init__(self):
        self.orders = []
    
    def add_order(self, order: Order):
        self.orders.append(order)
    
    def get_orders(self):
        return self.orders
     
     
        
