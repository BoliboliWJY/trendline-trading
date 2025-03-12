class OrderManager:
    """
    管理所有订单的类，用于存放未完成订单和已完成订单归档

    属性:
        pending_orders (dict): 存储待完成的订单，键为 order_id，值为 OrderBook 对象
        completed_orders (dict): 存储已完成的订单，键为 order_id，值为 OrderBook 对象
    """
    def __init__(self):
        self.pending_orders = {}
        self.completed_orders = {}

    def add_order(self, order):
        """
        添加新的订单

        参数:
            order: OrderBook 对象
        """
        if order.order_id in self.pending_orders or order.order_id in self.completed_orders:
            raise Exception(f"订单 {order.order_id} 已经存在")
        self.pending_orders[order.order_id] = order

    def complete_order(self, order_id):
        """
        归档完成订单，将订单从待完成订单中移除，并归到已完成订单列表中

        参数:
            order_id: 订单编号
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            if order.status != "closed":
                raise Exception(f"订单 {order_id} 状态异常，尚未平仓")
            self.completed_orders[order_id] = order
        else:
            raise Exception(f"订单 {order_id} 不存在")

    def get_pending_orders(self):
        """
        获取所有尚未完成的订单

        返回:
            list: 包含所有待完成订单的 OrderBook 对象列表
        """
        return list(self.pending_orders.values())

    def get_order(self, order_id):
        """
        根据订单编号查询订单

        参数:
            order_id (str/int): 订单编号

        返回:
            OrderBook: 目标订单对象（无论是未完成还是已完成状态）
        """
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        else:
            raise Exception(f"订单 {order_id} 不存在")
    