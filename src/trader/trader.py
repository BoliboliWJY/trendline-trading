import numpy as np
from src.utils import profile_method

# @profile_method
class Trader:
    def __init__(self,data:np.ndarray, trend_config:dict, trading_config:dict):
        self.data = data
        self.trend_config = trend_config
        self.trading_config = trading_config
        
        self.current_highest = [[],[]]
        self.current_lowest = [[],[]]
    
    
    def calculate_trend_price(self, data:np.ndarray, base_trend_number:int, current_trend:dict):
        """
        计算趋势价格（优化版：避免切片复制）
        """
        self.delay = self.trend_config["delay"] - 2
        self.data = data
        self.current_tick = self.data[base_trend_number, -1]
        self.trend_high = current_trend["trend_high"]
        self.trend_low = current_trend["trend_low"]
        
        self.deleted_high = current_trend["deleted_high"]
        n_high = len(current_trend["trend_high"])
        min_slope = np.inf
        
        # HACK 这段代码很可能有问题，但是目前暂未发现，所以暂时保留，如果什么发现无法找到最近的点，就遗弃使用下方注释代码
        new_entries = [[idx, self.trend_high[j][0]]
                       for idx, (slope, j) in self.deleted_high
                       if self.trend_high[j]]
        # 一次性将新数据追加到 deleted_high 中
        self.deleted_high.extend(new_entries)
        if (not self.current_highest[0]) or (self.current_highest[0] < base_trend_number - self.delay) or (self.data[self.current_highest[0], 1] < self.data[base_trend_number, 1]):
            self.current_highest = [[],[]] # 如果当前最高点已经超过delay或者有更大的点出现，则清空
        if self.current_highest == [[],[]]: # 如果当前最高点为空，则更新
            self.current_highest[0] = base_trend_number
            for real_index in range(n_high - 1, -1, -1):
                x = current_trend["trend_high"][real_index]
                if not x:
                    continue
                if x[0][0] < min_slope:
                    min_slope = x[0][0]
                    
                    for value in x:
                        self.deleted_high.append([real_index, value])
                        self.current_highest[1].append([real_index, value])

                
                if x[0][0] < 0:
                    break
        else:
            for i in range(n_high - 1, n_high - 1 - self.delay, -1):
                if current_trend["trend_high"][i]:
                    self.deleted_high.append((i, current_trend["trend_high"][i][0]))
            self.deleted_high.extend(self.current_highest[1])
            
        

            
        



        self.deleted_low = current_trend["deleted_low"]
        n_low = len(current_trend["trend_low"])
        min_slope = -np.inf
        
        new_entries = [[idx, self.trend_low[j][-1]] 
                       for idx, (slope, j) in self.deleted_low 
                       if self.trend_low[j]]
        # 一次性将新数据追加到 deleted_low 中
        self.deleted_low.extend(new_entries)
        if (not self.current_lowest[0]) or (self.current_lowest[0] < base_trend_number - self.delay) or (self.data[self.current_lowest[0], 3] > self.data[base_trend_number, 3]):
            self.current_lowest = [[],[]]
        if self.current_lowest == [[],[]]:

            
            self.current_lowest[0] = base_trend_number
            for real_index in range(n_low - 1, -1, -1):
                x = current_trend["trend_low"][real_index]
                if not x:
                    continue
                if x[-1][0] > min_slope:
                    min_slope = x[-1][0]
                    for value in x:
                        self.deleted_low.append([real_index, value])
                        self.current_lowest[1].append([real_index, value])
                if x[-1][0] > 0:
                    break
                
                
        else:
            for i in range(n_low - 1, n_low - 1 - self.delay, -1):
                if current_trend["trend_low"][i]:
                    self.deleted_low.append([i, current_trend["trend_low"][i][-1]])
            self.deleted_low.extend(self.current_lowest[1])
            
    
        

        
        self.trend_price = self.calculate_trend_price_by_trend()
        # 计算趋势价格
        
        return self.trend_price
        
        
    def calculate_trend_price_by_trend(self):
        """使用矢量化计算趋势价格（优化建议）"""
        # 计算高点趋势价格
        if self.deleted_high:
            # 使用列表推导式分离斜率和起始点，稍作重构使变量命名更清晰
            slopes_high = np.array([entry[1][0] for entry in self.deleted_high])
            start_indices_high = np.array([entry[1][1] for entry in self.deleted_high])
            # 预计算时间差，避免重复计算
            dt_high = self.current_tick - self.data[start_indices_high, 0]
            price_from_start = self.data[start_indices_high, 1]
            trend_price_high = np.column_stack((
                np.full(slopes_high.shape, self.current_tick),
                price_from_start + slopes_high * dt_high
            ))
            trend_price_high = np.sort(trend_price_high, axis = 0)
        else:
            trend_price_high = np.empty((0, 2))
            
        # if trend_price_high.size:
        #     # np.unique按整行去重，因为第一列均为当前tick，所以实际上只关注第二列价格是否相同
        #     unique_high = np.unique(trend_price_high, axis=0)
        #     order_high = np.argsort(unique_high[:, 1])
        #     trend_price_high = unique_high[order_high]

        # 计算低点趋势价格
        if self.deleted_low:
            slopes_low = np.array([entry[1][0] for entry in self.deleted_low])
            start_indices_low = np.array([entry[1][1] for entry in self.deleted_low])
            dt_low = self.current_tick - self.data[start_indices_low, 2]
            price_from_start = self.data[start_indices_low, 3]
            trend_price_low = np.column_stack((
                np.full(slopes_low.shape, self.current_tick),
                price_from_start + slopes_low * dt_low
            ))
            trend_price_low = np.sort(trend_price_low, axis=0)[::-1, :]
        else:
            trend_price_low = np.empty((0, 2))
            
        # if trend_price_low.size:
        #     unique_low = np.unique(trend_price_low, axis=0)
        #     order_low = np.argsort(unique_low[:, 1])[::-1] 
        #     trend_price_low = unique_low[order_low]

        return {"trend_price_high": trend_price_high, "trend_price_low": trend_price_low}
    
    
    
