import itertools
import numpy as np

class TrendTickCalculator:
    def __init__(self, data: np.ndarray, trend_config: dict, initial_filtered_trend_data: dict):
        self.data = data
        self.trend_config = trend_config
        self.trend_high = initial_filtered_trend_data["trend_high"]
        self.trend_low = initial_filtered_trend_data["trend_low"]
        
        # 扁平化趋势数据
        self.trend_high = set(
            tuple(item)
            for item in itertools.chain.from_iterable(sub for sub in self.trend_high if sub)
        )
        self.trend_low = set(
            tuple(item)
            for item in itertools.chain.from_iterable(sub for sub in self.trend_low if sub)
        )
        
        # 使用字典缓存初始截距，键为趋势 (斜率, 索引)
        self.trend_high_intercepts_dict = {}
        for trend in self.trend_high:
            k = trend[0]
            idx = int(trend[1])
            x1 = self.data[idx, 0]
            y1 = self.data[idx, 1]
            b = y1 - k * x1
            self.trend_high_intercepts_dict[trend] = (k, b)

        self.trend_low_intercepts_dict = {}
        for trend in self.trend_low:
            k = trend[0]
            idx = int(trend[1])
            x1 = self.data[idx, 2]
            y1 = self.data[idx, 3]
            b = y1 - k * x1
            self.trend_low_intercepts_dict[trend] = (k, b)

        
    def compute_intercepts_vectorized(self, trends_set, data:np.ndarray, col_start:int):
        if not trends_set:
            return np.empty((0, 2))
        trend_arr = np.array(list(trends_set))
        slopes = trend_arr[:, 0]
        indices = trend_arr[:, 1].astype(int)
        x1 = data[indices, col_start]
        y1 = data[indices, col_start + 1]
        intercepts = y1 - slopes * x1
        return np.column_stack((slopes, intercepts))
        
    def update_trend_data(self, data:np.ndarray, base_trend_number:int, current_trend:dict):
        """更新趋势数据"""
        # 新增项
        new_trend_high_set = {tuple(item) for item in current_trend["trend_high"][-1]}
        if new_trend_high_set:
            for trend in new_trend_high_set:
                if trend not in self.trend_high:
                    self.trend_high.add(trend)
                    # 计算新增项的截距（col_start 对于高点为 0）
                    k = trend[0]
                    idx = int(trend[1])
                    x1 = data[idx, 0]
                    y1 = data[idx, 1]
                    b = y1 - k * x1
                    self.trend_high_intercepts_dict[trend] = (k, b)

        # 新增项：低点趋势
        new_trend_low_set = {tuple(item) for item in current_trend["trend_low"][-1]}
        if new_trend_low_set:
            for trend in new_trend_low_set:
                if trend not in self.trend_low:
                    self.trend_low.add(trend)
                    # 计算新增项的截距（col_start 对于低点为 2）
                    k = trend[0]
                    idx = int(trend[1])
                    x1 = data[idx, 2]
                    y1 = data[idx, 3]
                    b = y1 - k * x1
                    self.trend_low_intercepts_dict[trend] = (k, b)

        # 删除项：高点趋势
        deleted_high = [tuple(item[1]) for item in current_trend["deleted_high"]]
        for trend in deleted_high:
            if trend in self.trend_high:
                self.trend_high.discard(trend)
            if trend in self.trend_high_intercepts_dict:
                del self.trend_high_intercepts_dict[trend]

        # 删除项：低点趋势
        deleted_low = [tuple(item[1]) for item in current_trend["deleted_low"]]
        for trend in deleted_low:
            if trend in self.trend_low:
                self.trend_low.discard(trend)
            if trend in self.trend_low_intercepts_dict:
                del self.trend_low_intercepts_dict[trend]

        # 如果后续需要使用 NumPy 数组形式的截距，可以只转换变化后的部分
        self.trend_high_intercepts = np.array(list(self.trend_high_intercepts_dict.values()))
        self.trend_low_intercepts = np.array(list(self.trend_low_intercepts_dict.values()))
        
        self.trend_price_high = self.calculate_trend_klines_price(data, base_trend_number, self.trend_high_intercepts)
        self.trend_price_low = self.calculate_trend_klines_price(data, base_trend_number, self.trend_low_intercepts)
        
        # 对 high 趋势价格按照第二列（价格）进行升序排序
        sorted_trend_price_high = self.trend_price_high[self.trend_price_high[:, 1].argsort()]
        # 对 low 趋势价格按照第二列（价格）进行降序排序
        sorted_trend_price_low = self.trend_price_low[self.trend_price_low[:, 1].argsort()[::-1]]
        return {"trend_price_high": sorted_trend_price_high, "trend_price_low": sorted_trend_price_low}
        
        
        
    def calculate_trend_klines_price(self, data, base_trend_number, trend_intercepts):
        """计算对应k线价格
        """
        if trend_intercepts.shape[0] == 0:
            return np.empty((0, 2))
        x = data[base_trend_number, -1]
        y = trend_intercepts[:,0] * x + trend_intercepts[:,1]
        x = np.full(y.shape, x, dtype=float)
        return np.column_stack((x, y))
        
        
        
    # def calculate_trend_price(self, data:np.ndarray, base_trend_number:int, current_trend:dict):
    #     """
    #     计算趋势价格（优化版：避免切片复制）
    #     """
    #     self.delay = self.trend_config["delay"] - 2
    #     self.data = data
    #     self.current_tick = self.data[base_trend_number, -1]
    #     self.trend_high = current_trend["trend_high"]
    #     self.trend_low = current_trend["trend_low"]
        
    #     self.deleted_high = current_trend["deleted_high"]
    #     self.remained_trend_high = []
    #     n_high = len(current_trend["trend_high"])
    #     min_slope = np.inf
        
    #     # HACK 这段代码很可能有问题，但是目前暂未发现，所以暂时保留，如果什么发现无法找到最近的点，就遗弃使用下方注释代码
    #     if (not self.current_highest[0]) or (self.current_highest[0] < base_trend_number - self.delay) or (self.data[self.current_highest[0], 1] < self.data[base_trend_number, 1]):
    #         self.current_highest = [[],[]] # 如果当前最高点已经超过delay或者有更大的点出现，则清空
    #     if self.current_highest == [[],[]]: # 如果当前最高点为空，则更新
    #         self.current_highest[0] = base_trend_number
    #         for i in self.remained_trend_high_last:
    #                 if i not in self.deleted_high:
    #                     self.remained_trend_high.append(i)
    #         for real_index in range(n_high - 1, -1, -1):
    #             x = current_trend["trend_high"][real_index]
    #             if not x:
    #                 continue
    #             if x[0][0] < min_slope:
    #                 min_slope = x[0][0]
                    
    #                 for value in x:
    #                     self.remained_trend_high.append([real_index, value])
    #                     self.current_highest[1].append([real_index, value])

                
                        
    #             # for i in self.remained_trend_high:
    #             #     if i != [] and i in self.deleted_high_last:
    #             #         self.remained_trend_high.remove(i)
    #             if x[0][0] < 0:
    #                 break
    #     else:
    #         for i in range(n_high - 1, n_high - 1 - self.delay, -1):
    #             if current_trend["trend_high"][i]:
    #                 self.remained_trend_high.append((i, current_trend["trend_high"][i][0]))
    #         self.remained_trend_high.extend(self.current_highest[1])
        
    #     if self.remained_trend_high_last != []:
    #         for i in [items[1][1] for items in self.remained_trend_high_last]:
    #             for j in self.trend_high[i]:
    #                 if j != [] and j not in self.deleted_high:
    #                     self.remained_trend_high.append((i, j))
        
    #     for i in [items[1][1] for items in self.deleted_high]:
    #         for j in self.trend_high[i]:
    #             if j != []:
    #                 self.remained_trend_high.append([i, j])
    #     self.remained_trend_high.extend(self.deleted_high)
        
    #     self.remained_trend_high_last = self.remained_trend_high
    #     self.deleted_high_last = self.deleted_high




    #     self.deleted_low = current_trend["deleted_low"]
    #     self.remained_trend_low = []
    #     n_low = len(current_trend["trend_low"])
    #     min_slope = -np.inf
        
    #     if (not self.current_lowest[0]) or (self.current_lowest[0] < base_trend_number - self.delay) or (self.data[self.current_lowest[0], 3] > self.data[base_trend_number, 3]):
    #         self.current_lowest = [[],[]]
    #     if self.current_lowest == [[],[]]:
    #         self.current_lowest[0] = base_trend_number
    #         for i in self.remained_trend_low_last:
    #                 if i != [] and i not in self.deleted_low:
    #                     self.remained_trend_low.append(i)
    #         for real_index in range(n_low - 1, -1, -1):
    #             x = current_trend["trend_low"][real_index]
    #             if not x:
    #                 continue
    #             if x[-1][0] > min_slope:
    #                 min_slope = x[-1][0]
                    
    #                 for value in x:
    #                     self.remained_trend_low.append([real_index, value])
    #                     self.current_lowest[1].append([real_index, value])
                        
                
                        
    #             if x[-1][0] > 0:
    #                 break
                
                
    #     else:
    #         for i in range(n_low - 1, n_low - 1 - self.delay, -1):
    #             if current_trend["trend_low"][i]:
    #                 self.remained_trend_low.append([i, current_trend["trend_low"][i][-1]])
    #         self.remained_trend_low.extend(self.current_lowest[1])
            
    #     if self.remained_trend_low_last != []:
    #         for i in [items[1][1] for items in self.remained_trend_low_last]:
    #             for j in self.trend_low[i]:
    #                 if j != [] and j not in self.deleted_low:
    #                     self.remained_trend_low.append([i, j])
                        
    #     for i in [items[1][1] for items in self.deleted_low]:
    #         for j in self.trend_low[i]:
    #             if j != []:
    #                 self.remained_trend_low.append([i, j])
    #     self.remained_trend_low.extend(self.deleted_low)
        
    #     self.remained_trend_low_last = self.remained_trend_low
    #     self.deleted_low_last = self.deleted_low
            
            
            
    #     self.trend_price = self.calculate_trend_price_by_trend()
        
    #     return self.trend_price
        
        
        
    # # def calculate_trend_price_by_trend(self):
    # #     """使用矢量化计算趋势价格（优化建议）"""
    # #     # 计算高点趋势价格
    # #     if self.deleted_high:
    # #         # 使用列表推导式分离斜率和起始点，稍作重构使变量命名更清晰
    # #         slopes_high = np.array([entry[1][0] for entry in self.deleted_high])
    # #         start_indices_high = np.array([entry[1][1] for entry in self.deleted_high])
    # #         # 预计算时间差，避免重复计算
    # #         dt_high = self.current_tick - self.data[start_indices_high, 0]
    # #         price_from_start = self.data[start_indices_high, 1]
    # #         trend_price_high = np.column_stack((
    # #             np.full(slopes_high.shape, self.current_tick),
    # #             price_from_start + slopes_high * dt_high
    # #         ))
    # #         trend_price_high = np.vstack([trend_price_high, np.array([[self.current_tick, np.inf]])])
    # #         trend_price_high = np.sort(trend_price_high, axis = 0)
    # #     else:
    # #         trend_price_high = np.array([[self.current_tick, np.inf]])
            
    # #     # if trend_price_high.size:
    # #     #     # np.unique按整行去重，因为第一列均为当前tick，所以实际上只关注第二列价格是否相同
    # #     #     unique_high = np.unique(trend_price_high, axis=0)
    # #     #     order_high = np.argsort(unique_high[:, 1])
    # #     #     trend_price_high = unique_high[order_high]

    # #     # 计算低点趋势价格
    # #     if self.deleted_low:
    # #         slopes_low = np.array([entry[1][0] for entry in self.deleted_low])
    # #         start_indices_low = np.array([entry[1][1] for entry in self.deleted_low])
    # #         dt_low = self.current_tick - self.data[start_indices_low, 2]
    # #         price_from_start = self.data[start_indices_low, 3]
    # #         trend_price_low = np.column_stack((
    # #             np.full(slopes_low.shape, self.current_tick),
    # #             price_from_start + slopes_low * dt_low
    # #             ))
    # #         trend_price_low = np.vstack([trend_price_low, np.array([[self.current_tick, -np.inf]])])
    # #         trend_price_low = np.sort(trend_price_low, axis=0)[::-1, :]
    # #     else:
    # #         trend_price_low = np.array([[self.current_tick, -np.inf]])
            
    # #     # if trend_price_low.size:
    # #     #     unique_low = np.unique(trend_price_low, axis=0)
    # #     #     order_low = np.argsort(unique_low[:, 1])[::-1] 
    # #     #     trend_price_low = unique_low[order_low]
        
    # #     # 整理趋势线数据（去除重复数据）
    # #     trend_high = [[entry[1][0], self.data[entry[1][1], 0]] for entry in self.deleted_high]
    # #     trend_low = [[entry[1][0], self.data[entry[1][1], 2]] for entry in self.deleted_low]
    # #     # trend_high = np.unique(np.array(trend_high), axis=0)
    # #     # trend_low = np.unique(np.array(trend_low), axis=0)

    # #     return {"trend_price_high": trend_price_high, "trend_price_low": trend_price_low, "trend_high": trend_high, "trend_low": trend_low}

    # def calculate_trend_price_by_trend(self):
    #     """结合矢量化计算趋势价格与遍历过滤趋势线数据，避免使用 np.unique 提高效率

    #     返回的数据中包含：
    #         - trend_price_high, trend_price_low：根据当前 tick 及 deleted 数据计算得到的趋势价格；
    #         - trend_high, trend_low：在计算过程中手动过滤重复后的趋势数据。
    #     """
    #     # 计算高点趋势价格
    #     if self.remained_trend_high:
    #         slopes_high = np.array([entry[1][0] for entry in self.remained_trend_high])
    #         start_indices_high = np.array([entry[1][1] for entry in self.remained_trend_high])
    #         dt_high = self.current_tick - self.data[start_indices_high, 0]
    #         price_from_start = self.data[start_indices_high, 1]
    #         trend_price_high = np.column_stack((
    #             np.full(slopes_high.shape, self.current_tick),
    #             price_from_start + slopes_high * dt_high
    #         ))
    #         # 添加一个代表无穷大的边界点，并排序（注意这里的排序方式根据实际需求调整）
    #         trend_price_high = np.vstack([trend_price_high, np.array([[self.current_tick, np.inf]])])
    #         trend_price_high = np.sort(trend_price_high, axis=0)
    #     else:
    #         trend_price_high = np.array([[self.current_tick, np.inf]])
        
    #     # 计算低点趋势价格
    #     if self.remained_trend_low:
    #         slopes_low = np.array([entry[1][0] for entry in self.remained_trend_low])
    #         start_indices_low = np.array([entry[1][1] for entry in self.remained_trend_low])
    #         dt_low = self.current_tick - self.data[start_indices_low, 2]
    #         price_from_start = self.data[start_indices_low, 3]
    #         trend_price_low = np.column_stack((
    #             np.full(slopes_low.shape, self.current_tick),
    #             price_from_start + slopes_low * dt_low
    #         ))
    #         trend_price_low = np.vstack([trend_price_low, np.array([[self.current_tick, -np.inf]])])
    #         trend_price_low = np.sort(trend_price_low, axis=0)[::-1, :]
    #     else:
    #         trend_price_low = np.array([[self.current_tick, -np.inf]])
        
    #     # 在计算过程中手动过滤重复的趋势数据
    #     # 高点趋势数据：每个记录为 (斜率, 对应 self.data[索引, 0])
    #     trend_high_list = []
    #     for entry in self.remained_trend_high:
    #         k = entry[1][0]
    #         idx = entry[1][1]
    #         x1 = self.data[idx, 0]
    #         y1 = self.data[idx, 1]
    #         b = y1 - k * x1
    #         current_item = (k, b)
    #         # 如果列表为空或与上一次记录不同则保存
    #         if not trend_high_list or trend_high_list[-1] != current_item:
    #             trend_high_list.append(current_item)
    #     trend_high = np.array(trend_high_list)
        
    #     # 低点趋势数据：每个记录为 (斜率, 对应 self.data[索引, 2])
    #     trend_low_list = []
    #     for entry in self.remained_trend_low:
    #         k = entry[1][0]
    #         idx = entry[1][1]
    #         x1 = self.data[idx, 2]
    #         y1 = self.data[idx, 3]
    #         b = y1 - k * x1
    #         current_item = (k, b)
    #         # 如果列表为空或与上一次记录不同则保存
    #         if not trend_low_list or trend_low_list[-1] != current_item:
    #             trend_low_list.append(current_item)
    #     trend_low = np.array(trend_low_list)
        
    #     return {
    #         "trend_price_high": trend_price_high,
    #         "trend_price_low": trend_price_low,
    #         "trend_high": trend_high,
    #         "trend_low": trend_low,
    #     }