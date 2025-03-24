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
        
    def update_trend_data(self, data:np.ndarray, current_trend:dict, current_time_high:int, current_time_low = None):
        """更新趋势数据"""
        if current_time_low is None: # 判断是实时还是回测情况
            current_time_high = current_time_high
            current_time_low = None
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
        
        if current_time_low is None:
            self.trend_price_high = self.calculate_trend_klines_price(data, current_time_high, self.trend_high_intercepts)
            self.trend_price_low = self.calculate_trend_klines_price(data, current_time_high, self.trend_low_intercepts)
        else:
            self.trend_price_high = self.calculate_trend_klines_price(data, current_time_high, self.trend_high_intercepts)
            self.trend_price_low = self.calculate_trend_klines_price(data, current_time_low, self.trend_low_intercepts)
        
        # 对 high 趋势价格按照第二列（价格）进行升序排序
        sorted_trend_price_high = self.trend_price_high[self.trend_price_high[:, 1].argsort()]
        # 对 low 趋势价格按照第二列（价格）进行降序排序
        sorted_trend_price_low = self.trend_price_low[self.trend_price_low[:, 1].argsort()[::-1]]
        return {"trend_price_high": sorted_trend_price_high, "trend_price_low": sorted_trend_price_low}
        
        
        
    def calculate_trend_klines_price(self, data, current_time, trend_intercepts):
        """计算对应k线价格
        """
        if trend_intercepts.shape[0] == 0:
            return np.empty((0, 2))
        x = current_time
        y = trend_intercepts[:,0] * x + trend_intercepts[:,1]
        x = np.full(y.shape, x, dtype=float)
        return np.column_stack((x, y))