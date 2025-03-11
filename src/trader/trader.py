import numpy as np
from src.utils import profile_method
from src.order_book.order_booker import Order
# @profile_method
class Trader:
    def __init__(self,data:np.ndarray, trend_config:dict, trading_config:dict):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        
        self.buy_open_potential_signal = False # 潜在买入开仓信号
        self.sell_open_potential_signal = False # 潜在卖出开仓信号
        self.buy_open_potential_signal_last = False # 上一根k线的潜在买入开仓信号
        self.sell_open_potential_signal_last = False # 上一根k线的潜在卖出开仓信号
        
        self.reserved_buy_open_signal = False # 保留买入开仓信号
        self.reserved_sell_open_signal = False # 保留卖出开仓信号
        
        self.high_signal_found = False # 高趋势线信号找到
        self.low_signal_found = False # 低趋势线信号找到
        
        self.trend_price_high_last = 0 # 上一次趋势线最近的高点
        self.trend_price_low_last = 0 # 上一次趋势线最近的低点
    
        self.open_signals = {"high_open":[], "low_open":[], "high_open_enter":[], "low_open_enter":[], "sell_close_ideal":[], "buy_close_ideal":[]}
        self.close_signals = {"high_close":[], "low_close":[]}
        
        self.close_potential_signal = False # 潜在平仓信号，需要具体对tick分析
        
        self.order_book = []
        
        self.paused = False # 是否暂停可视化更新
        
    def open_close_signal(self,max_min_price:np.ndarray, trend_price:np.ndarray, price_time_array:np.ndarray, base_trend_number:int):
        """
        根据趋势价格和价格时间数组，生成开仓和平仓信号
        """
        # trend_price_high = trend_price["trend_price_high"]
        # trend_price_low = trend_price["trend_price_low"]
        # trend_high = trend_price["trend_high"] # 趋势线斜率，截距
        # trend_low = trend_price["trend_low"]
        trend_price_high = trend_price["trend_price_high"][:, 1] # 对应k线下的价格
        trend_price_low = trend_price["trend_price_low"][:, 1]
        price_time_array = price_time_array
        time = price_time_array[:, 0]
        tick_price = price_time_array[:, 1]
        
        self.base_trend_number = base_trend_number # 当前k线对应data索引
        
        max_price = np.max(tick_price)
        max_index = np.argmax(tick_price)
        max_after_min = np.min(tick_price[max_index:])
        min_price = np.min(tick_price)
        min_index = np.argmin(tick_price)
        min_after_max = np.max(tick_price[min_index:])
        start_price = tick_price[0]
        end_price = tick_price[-1]
        
        self.paused = False # 重置暂停状态
        
        # trend_price_high_tick = self.compute_trend_price(trend_high, time) # 对应tick下的价格
        # trend_price_low_tick = self.compute_trend_price(trend_low, time)
        
        # if not self.open_potential_signal: # 基于k线价格简单判断
        # 做空进入阈值
        sell_enter_val = 1 - max_price / trend_price_high[0] if trend_price_high.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做空潜在利润值
        sell_profit_val = 1 - trend_price_low[0] / max_price if trend_price_low.size > 0 else 1
        # 做多进入阈值
        buy_enter_val = 1 - trend_price_low[0] / min_price if trend_price_low.size > 0 else self.trading_config["enter_threshold"] + 1
        # 做多潜在利润值
        buy_profit_val = 1 - min_price / trend_price_high[0] if trend_price_high.size > 0 else 1
        
        if sell_enter_val < self.trading_config["enter_threshold"] and sell_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在卖出开仓信号, 进入阈值/打破趋势线")
            # print("最高价格", self.max_price, "趋势价格", trend_price_high[0, 1], "相差", sell_enter_val)
            # self.paused = True # 暂停可视化
            self.sell_open_potential_signal = True # 出现潜在卖出开仓信号
            # self.buy_open_potential_signal = False # 重置潜在买入开仓信号
            # self.sell_open_potential_signal_last = True
            # self.buy_open_potential_signal_last = False
        elif buy_enter_val < self.trading_config["enter_threshold"] and buy_profit_val > self.trading_config["potential_profit"]:
            # print("出现潜在买入开仓信号, 进入阈值/打破趋势线")
            # print("最低价格", self.min_price, "趋势价格", trend_price_low[0, 1], "相差", buy_enter_val)
            # self.paused = True # 暂停可视化
            self.buy_open_potential_signal = True # 出现潜在买入开仓信号
            # self.sell_open_potential_signal = False # 重置潜在卖出开仓信号
            # self.buy_open_potential_signal_last = True
            # self.sell_open_potential_signal_last = False
        else: # 如果啥信号也没有就重置，避免影响非连续的点
            self.paused = False
            self.buy_open_potential_signal = False
            self.sell_open_potential_signal = False
            # self.buy_open_potential_signal_last = False
            # self.sell_open_potential_signal_last = False
        

        if self.sell_open_potential_signal or self.sell_open_potential_signal_last: # 如果出现潜在卖出开仓信号
            self.process_open_signal(trend_price_high, tick_price, self.sell_open_potential_signal_last, time, "sell", "high_open", self.high_signal_found)
            
        if self.buy_open_potential_signal or self.buy_open_potential_signal_last: # 如果出现潜在买入开仓信号
            self.process_open_signal(trend_price_low, tick_price, self.buy_open_potential_signal_last, time, "buy", "low_open", self.low_signal_found)
            
            
    
    
    def process_open_signal(self, trend_prices, tick_prices, last_signal, time, order_type:str, open_signal_key, signal_found):
        """
        合并后的处理开仓信号函数，可用于处理买入和卖出开仓信号

        参数说明：
            trend_prices: 趋势价格数组（对应高或低趋势）
            tick_prices: tick价格数组
            last_signal: 上一根K线是否进入候选区域
            time: 时间数组
            direction: "high" 或者 "low"，区分对应趋势方向
            order_type: "sell" 或 "buy"，对应订单类型
            open_signal_key: 保存信号的键名，例如 "high_open" 或 "low_open"
            signal_found: 对应方向的信号标志，如 self.high_signal_found 或 self.low_signal_found
        """
        start_index = 0
        trend_idx = 0
        # self.open_signals[open_signal_key] = []
        while start_index < len(tick_prices):
            if not last_signal:
                enter_signal, start_index, trend_idx = self.judge_signal(trend_prices, tick_prices, start_index, trend_idx, signal_found, order_type, mode="enter")
                if start_index is None or trend_idx is None:
                    last_signal = False # 没有进入趋势线，且循环结束
                    break
                if enter_signal: # 进入阈值
                    # self.paused = True # 进入阈值，绘图
                    current_time = time[start_index]
                    current_price = tick_prices[start_index]
                    self.open_signals[open_signal_key + "_enter"].append([current_time, current_price])
                    # if order_type == "sell": # 如果出现方向进入阈值信号，则重置相反方向的开仓信号，因为可能出现潜在ya
                    #     self.buy_open_potential_signal_last = False # 重置买入开仓信号
                    # elif order_type == "buy":
                    #     self.sell_open_potential_signal_last = False # 重置卖出开仓信号

            leave_signal, start_index, trend_idx = self.judge_signal(trend_prices, tick_prices, start_index, trend_idx, signal_found, order_type, mode="leave")
            if not leave_signal and start_index is not None: # 打破阈值但未循环结束
                last_signal = False
                continue
            if start_index is None or trend_idx is None:
                last_signal = True # 没有开仓，但是已经进入过趋势线
                break
            if leave_signal: # 执行开仓操作
                self.paused = True # 出现开仓信号，绘图
                current_time = time[start_index]
                current_price = tick_prices[start_index]
                # 下单操作与记录
                
                self.open_signals[open_signal_key].append([current_time, current_price])
                if order_type == "sell":
                    self.sell_open_potential_signal_last = False
                    stop_loss, max_profit = self.predict_profit_loss(order_type, current_price, np.max(tick_prices[start_index:]))
                elif order_type == "buy":
                    self.buy_open_potential_signal_last = False
                    stop_loss, max_profit = self.predict_profit_loss(order_type, current_price, np.min(tick_prices[start_index:]))
                # print(f"是否止损: {stop_loss}, 最大回撤: {max_drawdown}, 最大收益: {max_profit}")
                
                self.place_order(order_type, current_price, current_time, stop_loss, max_profit)
                
                
                last_signal = False
                continue
        if order_type == "sell":
            self.sell_open_potential_signal_last = last_signal
        elif order_type == "buy":
            self.buy_open_potential_signal_last = last_signal
                
        
        
    def place_order(self, order_type:str, price:float, tick_time, stop_loss, profit):
        order_id = len(self.order_book) + 1
        order = {
            "order_id": order_id,
            "order_type": order_type,
            "price": price,
            "tick_time": tick_time,
            "status": "open",
            "stop_loss": stop_loss,
            "profit": profit,
        }
        self.order_book.append(order)
        # print(f"开仓订单: {order}")

    
        
    def judge_signal(self, trend_prices, tick_prices, start_index, trend_idx, signal_found, direction:str, mode = "enter"):
        """
        根据模式判断是否进入或离开信号区域
        mode: "enter" 表示判断进入候选区域；"leave" 表示判断离开后触发信号
        """
        # signal_found = False
        while start_index < len(tick_prices):
            if trend_idx >= len(trend_prices):
                break
            current_price = tick_prices[start_index]
            # ratio = 1 - current_price / trend_prices[trend_idx]
            if direction == "sell":
                ratio = (trend_prices[trend_idx] - current_price) / current_price
            else:
                ratio = (current_price - trend_prices[trend_idx]) / current_price
            if mode == "enter":
                if not signal_found and ratio < self.trading_config["enter_threshold"]:
                    signal_found = True
                    return True, start_index, trend_idx
                elif signal_found:
                    if ratio < 1e-5:
                        # BUG 因为没有计算趋势线的具体tick价格，所以这里需要一个很小的阈值
                        # 出现打破趋势价格的情况，更新趋势线下标并重置进入标志
                        trend_idx += 1
                        signal_found = False
                        return False, start_index, trend_idx # 打破阈值但未循环结束
                    

            elif mode == "leave":
                if ratio > self.trading_config["leave_threshold"]:
                    return True, start_index, trend_idx
                elif ratio < 2e-4:
                    # 出现打破趋势价格的情况，更新趋势线下标并重置进入标志
                    trend_idx += 1
                    signal_found = False
                    return False, start_index, trend_idx # 打破阈值但未循环结束
            start_index += 1
        return False, None, None # 啥也没找到
        
    def predict_profit_loss(self, order_type:str, price:float, key_price:float):
        """
        预测未来k线结果出现止损、止盈或盈利情况
        Args:
            order_type: "sell" 或 "buy"，对应订单类型
            price: 当前价格
            key_price: 用于计算收益的关键价格
        Returns:
            (stop_loss, max_profit_before_stop_loss)
        """
        further_sight = self.trading_config["further_sight"]
        if self.base_trend_number + further_sight > len(self.data):
            further_sight = len(self.data) - self.base_trend_number - 1
        
        # 未来 k 线数据
        further_sight_data = self.data[self.base_trend_number + 1:self.base_trend_number + further_sight + 1, :]
        
        stop_loss = False  # 是否触发止损
        ideal_price = key_price
        ideal_time = self.data[self.base_trend_number][0]
        
        if order_type == "sell":
            # 卖出时收益计算： (开仓价格 - 关键价格) / 开仓价格
            max_profit_before_stop_loss = (price - key_price) / price
            if max_profit_before_stop_loss < -self.trading_config["trailing_stop_loss"]:
                stop_loss = True
                self.open_signals["sell_close_ideal"].append([ideal_time, key_price])
                return stop_loss, max(max_profit_before_stop_loss, -self.trading_config["trailing_stop_loss"])
            
            for i in range(further_sight):
                # 计算止损跌幅
                drawdown = (price - further_sight_data[i][1]) / price
                if drawdown < -self.trading_config["trailing_stop_loss"]:
                    stop_loss = True
                    self.open_signals["sell_close_ideal"].append([ideal_time, ideal_price])
                    return stop_loss, max_profit_before_stop_loss
                else:
                    # 计算最新的潜在收益（获利）
                    potential_profit = (price - further_sight_data[i][3]) / price
                    if max_profit_before_stop_loss < potential_profit:
                        max_profit_before_stop_loss = potential_profit
                        ideal_price = further_sight_data[i][3]
                        ideal_time = further_sight_data[i][2]
                        # 当获利达到 0.005 时触发止盈
                        if max_profit_before_stop_loss >= 0.005:
                            self.open_signals["sell_close_ideal"].append([ideal_time, ideal_price])
                            return stop_loss, max_profit_before_stop_loss
            self.open_signals["sell_close_ideal"].append([ideal_time, ideal_price])
                
        else:
            # 买入时收益计算： (关键价格 - 开仓价格) / 开仓价格
            max_profit_before_stop_loss = (key_price - price) / price
            if max_profit_before_stop_loss < -self.trading_config["trailing_stop_loss"]:
                stop_loss = True
                self.open_signals["buy_close_ideal"].append([ideal_time, key_price])
                return stop_loss, max(max_profit_before_stop_loss, -self.trading_config["trailing_stop_loss"])
            
            for i in range(further_sight):
                drawdown = (further_sight_data[i][3] - price) / price
                if drawdown < -self.trading_config["trailing_stop_loss"]:
                    stop_loss = True
                    self.open_signals["buy_close_ideal"].append([ideal_time, ideal_price])
                    return stop_loss, max_profit_before_stop_loss
                else:
                    potential_profit = (further_sight_data[i][1] - price) / price
                    if max_profit_before_stop_loss < potential_profit:
                        max_profit_before_stop_loss = potential_profit
                        ideal_price = further_sight_data[i][1]
                        ideal_time = further_sight_data[i][0]
                        # 当获利达到 0.006 时触发止盈
                        if max_profit_before_stop_loss >= 0.006:
                            self.open_signals["buy_close_ideal"].append([ideal_time, ideal_price])
                            return stop_loss, max_profit_before_stop_loss
            self.open_signals["buy_close_ideal"].append([ideal_time, ideal_price])
        
        return stop_loss, max_profit_before_stop_loss
        
        
        
        
        
        
        
    


    
            