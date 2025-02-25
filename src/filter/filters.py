import numpy as np

# !!!不用所有都迭代，毕竟每个斜率都是顺序排列的


class trend_filter:
    def __init__(self,data,config):
        self.data = data
        self.config = config

        # self.slope_records_high = [[] for _ in range(len(self.trend_high))]
        # self.slope_records_low = [[] for _ in range(len(self.trend_low))]
        # self.line_age_records_high = [[] for _ in range(len(self.trend_high))]
        # self.line_age_records_low = [[] for _ in range(len(self.trend_low))]
        # self.distance_records_high = [[] for _ in range(len(self.trend_high))]
        # self.distance_records_low = [[] for _ in range(len(self.trend_low))]
        # self.trending_line_records_high = [[] for _ in range(len(self.trend_high))]
        # self.trending_line_records_low = [[] for _ in range(len(self.trend_low))]

        # self.trend_high, self.trend_low = self.filter_trend_initial(self.trend_high, self.trend_low, data, config)

        # self.result = {"trend_high": self.trend_high, "trend_low": self.trend_low}

    def filter_trend_initial(self, trend_high, trend_low):
        """
        初始化过滤
        """
        enable_filter = self.config.get("enable_filter", True)
        if not enable_filter:
            return {"trend_high": trend_high, "trend_low": trend_low}

        # 反转限制
        self.delay = self.config.get("delay", 10) - 1
        trend_high = trend_high[: -self.delay]
        trend_low = trend_low[: -self.delay]

        if self.config.get("filter_reverse", False):
            trend_high = self.filter_by_reverse(trend_high, True)
            trend_low = self.filter_by_reverse(trend_low, False)

        # 斜率大小限制
        if self.config.get("filter_slope", False):
            trend_high = self.filter_by_slope(
                trend_high, threshold=self.config.get("slope_threshold", 1), is_high=True
            )
            trend_low = self.filter_by_slope(
                trend_low, threshold=self.config.get("slope_threshold", 1), is_high=False
            )

        # 最小生成间隔
        if self.config.get("filter_line_age", False):
            min_age_value = self.config.get("min_line_age", 5)
            trend_high = self.filter_by_line_age(
                trend_high, min_age=min_age_value, is_high=True
            )
            trend_low = self.filter_by_line_age(
                trend_low, min_age=min_age_value, is_high=False
            )

        # 最小距离
        if self.config.get("filter_distance", False):
            interval = self.config.get("interval", "1000*1000")
            trend_high = self.filter_by_distance(
                trend_high,
                self.data,
                distance_threshold=self.config.get("distance_threshold", 10),
                interval=interval,
                is_high=True,
            )
            trend_low = self.filter_by_distance(
                trend_low,
                self.data,
                distance_threshold=self.config.get("distance_threshold", 10),
                interval=interval,
                is_high=False,
            )

        if self.config.get("filter_trending_line", False):
            trend_high = self.filter_by_trending_line(trend_high, is_high=True)
            trend_low = self.filter_by_trending_line(trend_low, is_high=False)

        return {"trend_high": trend_high, "trend_low": trend_low}  # 返回初始过滤后的趋势数据

    def filter_by_reverse(self, trend, is_high):
        """过滤反转趋势"""
        filtered_trend = [[] for _ in range(len(trend))]

        # 预先过滤出需要的斜率
        for i in range(1, len(trend)):
            valid_slopes = [
                slope
                for slope, j in trend[i]
                if (is_high and slope < 0) or (not is_high and slope > 0)
            ]
            filtered_trend[i].extend(
                [[slope, j] for slope, j in trend[i] if slope in valid_slopes]
            )

        return filtered_trend

    def filter_by_slope(self, trend, threshold=1, is_high=True):
        """过滤斜率"""
        filtered_trend = [[] for _ in range(len(trend))]
        for i in range(1, len(trend)):
            for slope, j in trend[i]:
                if np.abs(slope) <= threshold:
                    filtered_trend[i].append([slope, j])
        return filtered_trend

    def filter_by_line_age(self, trend, min_age, is_high):
        """过滤趋势年龄"""
        filtered_trend = [[] for _ in range(len(trend))]

        for i in range(1, len(trend)):
            for slope, j in trend[i]:
                line_age = i - j  # Example calculation; adjust as needed
                if line_age >= min_age:
                    filtered_trend[i].append([slope, j])
        return filtered_trend

    def filter_by_distance(self, trend, data, distance_threshold, interval, is_high):
        """过滤距离"""
        filtered_trend = [[] for _ in range(len(trend))]

        for i in range(1, len(trend)):
            for slope, j in trend[i]:
                distance = (data[i, 0] - data[j, 0]) / interval
                if distance >= distance_threshold:
                    filtered_trend[i].append([slope, j])
        return filtered_trend

    def filter_by_trending_line(self, trend, is_high):
        """过滤处于趋势之中趋势数量小于2的线"""
        filtered_trend = [[] for _ in range(len(trend))]

        for i in range(1, len(trend)):
            for slope, j in trend[i]:
                if len(trend[j]) < 1:
                    continue
                filtered_trend[i].append([slope, j])
        return filtered_trend

    def filter_trend(
        self,
        original_trend_high,
        original_trend_low,
        trend_high,
        trend_low,
    ):
        """
        过滤趋势,只过滤最后一个元素
        Args:
            regional_trend_high: 原始高趋势
            regional_trend_low: 原始低趋势
            trend_high: 之前被过滤过的高趋势
            trend_low: 之前被过滤过的低趋势
            data: 数据
            config: 配置
        """
        # 添加新的趋势

        enable_filter = self.config.get("enable_filter", True)
        # delay = config.get("delay", 10)
        interval = self.config.get("interval", "1000000")

        # trend_high.append(list(original_trend_high[-delay]))
        # trend_low.append(list(original_trend_low[-delay]))

        if not enable_filter:
            return {
                "trend_high": self.filter_by_trending_line_last(trend_high, original_trend_high, 0),
                "trend_low": self.filter_by_trending_line_last(trend_low, original_trend_low, 0),
            }

        filters = {
            "filter_reverse": self.filter_by_reverse_last,
            "filter_slope": self.filter_by_slope_last,
            "filter_line_age": self.filter_by_line_age_last,
            "filter_distance": self.filter_by_distance_last,
            "filter_trending_line": self.filter_by_trending_line_last,
        }

        filter_reverse = self.config.get("filter_reverse", False)
        filter_slope = self.config.get("filter_slope", False)
        filter_line_age = self.config.get("filter_line_age", False)
        filter_distance = self.config.get("filter_distance", False)
        filter_trending_line = self.config.get("filter_trending_line", False)

        if not enable_filter:
            trend_high = self.filter_by_trending_line_last(
                trend_high,
                original_trend_high,
                0,
            )
            trend_low = self.filter_by_trending_line_last(
                trend_low, original_trend_low, 0
            )
            return trend_high, trend_low

        if filter_reverse:
            trend_high = self.filter_by_reverse_last(trend_high, True)
            trend_low = self.filter_by_reverse_last(trend_low, False)
        # 斜率大小限制
        if filter_slope:
            trend_high = self.filter_by_slope_last(
                trend_high, threshold=self.config.get("slope_threshold", 1)
            )
            trend_low = self.filter_by_slope_last(
                trend_low, threshold=self.config.get("slope_threshold", 1)
            )
        # 趋势年龄限制
        if filter_line_age:
            trend_high = self.filter_by_line_age_last(
                trend_high, min_age=self.config.get("min_line_age", 5)
            )
            trend_low = self.filter_by_line_age_last(
                trend_low, min_age=self.config.get("min_line_age", 5)
            )
        # 距离限制
        if filter_distance:
            trend_high = self.filter_by_distance_last(
                trend_high,
                self.data,
                distance_threshold=self.config.get("distance_threshold", 10),
                interval=interval,
            )
            trend_low = self.filter_by_distance_last(
                trend_low,
                self.data,
                distance_threshold=self.config.get("distance_threshold", 10),
                interval=interval,
            )
        # 趋势数量限制
        if filter_trending_line:
            trend_high = self.filter_by_trending_line_last(
                trend_high,
                original_trend_high,
                self.config.get("filter_trending_line_number", 5),
            )
            trend_low = self.filter_by_trending_line_last(
                trend_low,
                original_trend_low,
                self.config.get("filter_trending_line_number", 5),
            )
        return {"trend_high": trend_high, "trend_low": trend_low}

    def filter_by_reverse_last(self, trend, is_high):
        """过滤反转趋势"""
        new_last_row = []
        for slope, j in trend[-1]:
            if is_high:
                if slope >= 0:
                    continue
            else:
                if slope <= 0:
                    continue
            new_last_row.append([slope, j])
        trend[-1] = new_last_row
        return trend

    def filter_by_slope_last(self, trend, threshold=1):
        """Filter trends based on the absolute slope threshold."""
        new_last_row = []
        for slope, j in trend[-1]:
            if np.abs(slope) <= threshold:
                new_last_row.append([slope, j])
        trend[-1] = new_last_row
        return trend

    def filter_by_line_age_last(self, trend, min_age):
        """过滤趋势年龄"""
        new_last_row = []
        for slope, j in trend[-1]:
            line_age = len(trend) - j
            if line_age >= min_age:
                new_last_row.append([slope, j])
        trend[-1] = new_last_row
        return trend

    def filter_by_distance_last(self, trend, data, distance_threshold, interval):
        """过滤距离"""
        new_last_row = []
        for slope, j in trend[-1]:
            distance = (data[-1, 0] - data[j, 0]) / interval
            if distance >= distance_threshold:
                new_last_row.append([slope, j])
        trend[-1] = new_last_row
        return trend

    def filter_by_trending_line_last(self, trend, regional_trend, number):
        """过滤处于趋势之中趋势数量小于2的线"""
        new_last_row = []
        for slope, j in trend[-1]:
            if len(regional_trend[j]) < number:
                continue
            new_last_row.append([slope, j])
        trend[-1] = new_last_row
        return trend

    def process_new_trend(self, filtered_trend_data, current_trend):
        """处理新的趋势"""
        # 添加新的趋势
        if not self.config.get("enable_filter", False):
            return current_trend
        
        filtered_trend_data["trend_high"].append(
            list(current_trend["trend_high"][-self.config.get("delay", 10)])
        )
        filtered_trend_data["trend_low"].append(
            list(current_trend["trend_low"][-self.config.get("delay", 10)])
        )

        # 过滤趋势
        filtered_trend_data = self.filter_trend(
            current_trend["trend_high"],
            current_trend["trend_low"],
            filtered_trend_data["trend_high"],
            filtered_trend_data["trend_low"],
        )
        deleted_high = current_trend["deleted_high"]
        deleted_low = current_trend["deleted_low"]

        filtered_trend_high, removed_items_high = self._remove_items(
            filtered_trend_data["trend_high"], deleted_high
        )
        filtered_trend_low, removed_items_low = self._remove_items(
            filtered_trend_data["trend_low"], deleted_low
        )

        filtered_trend = {
            "trend_high": filtered_trend_high,
            "trend_low": filtered_trend_low,
        }
        return filtered_trend

    def _remove_items(self, filtered_list, deleted_items):
        removed_items = []
        for idx, item_to_delete in deleted_items:
            if idx < len(filtered_list):
                try:
                    filtered_list[idx].remove(item_to_delete)
                    removed_items.append([item_to_delete])
                except ValueError:
                    pass
        return filtered_list, removed_items
