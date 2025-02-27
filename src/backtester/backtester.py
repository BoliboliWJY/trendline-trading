import time
from src.utils import profile_method


# Import other dependencies from src as needed:
from src.trend_calculator.compute_initial_trends import compute_initial_trends
from src.trend_calculator.trend_generator import backtest_calculate_trend_generator

class Backtester:
    """历史数据回测器，但不再过滤趋势，直接输出未过滤的趋势以及每次被删除的趋势

    Returns:
        current_trend: 当前趋势数据
            trend_high: 完整的高趋势数据
            trend_low: 完整的低趋势数据
            deleted_high: 被删除的高趋势数据
            deleted_low: 被删除的低趋势数据
    """
    # @profile_method
    def __init__(
        self,
        data,
        type_data,
        base_trend_number=1000,
    ):
        self.data = data
        self.type_data = type_data
        self.trend_generator = backtest_calculate_trend_generator(data=data)
        self.base_trend_number = base_trend_number  # 前置趋势数量

        # 趋势管理
        # self.deleted_trends = {}
        self.trend_high = []  # 完整的高趋势数据
        self.trend_low = []  # 完整的低趋势数据

        # 可视化参数

        # 当前数据(随回测进行而增长)
        self.current_data = self.data[: self.base_trend_number]
        self.current_type = self.type_data[: self.base_trend_number]

        self.backtest_count = 1  # 回测计数

        # 初始化趋势，并保存返回的数据
        self.initial_trend_data = self.initial_trend()

    def initial_trend(self):
        """
        初始化趋势，并记录被删除的趋势
        """
        # 计算
        self.trend_high,self.trend_low = compute_initial_trends(self.current_data,self.trend_generator)
        # 保存计算结果
        # self.trend_high = trend_high
        # self.trend_low = trend_low

        # 返回结构化的趋势数据 初始化趋势
        return {
            "trend_high": self.trend_high,
            "trend_low": self.trend_low,
        }

    # @profile_method
    def update_trend(self):
        """更新一次趋势数据，并返回更新后的数据; 如果数据结束则返回 False"""
        end_index = self.base_trend_number + self.backtest_count
        if end_index >= len(self.data):
            print("Reached end of data. Stopping the backtest.")
            return False  # 如果数据结束，返回 False

        # 更新趋势数据
        try:
            self.trend_high, self.trend_low, deleted_high, deleted_low = next(
                self.trend_generator
            )
        except StopIteration:
            self.trend_high, self.trend_low, deleted_high, deleted_low = [], [], [], []

        updated_trend = {
            "trend_high": self.trend_high,
            "trend_low": self.trend_low,
            "deleted_high": deleted_high,
            "deleted_low": deleted_low,
        }

        self.backtest_count += 1 # 回测计数

        return updated_trend  # 返回更新后的趋势数据

    def _remove_items(self, filtered_list, deleted_items):
        removing_item = False
        removed_items = []
        for idx, item_to_delete in deleted_items:
            if idx < len(filtered_list):
                try:
                    filtered_list[idx].remove(item_to_delete)
                    removed_items.append([item_to_delete])
                    removing_item = True
                except ValueError:
                    pass
        return removing_item, removed_items

    # @profile_method
    def run_backtest(self, delay=0):
        """运行回测，同时可通过返回的趋势数据进行可视化"""
        while True:
            result = self.update_trend()
            if result is False:
                break
            if delay > 0:
                time.sleep(delay)
            yield result
