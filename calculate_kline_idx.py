from datetime import datetime, time, timedelta

def calculate_kline_idx(input_time_str):
    """
    计算从当月月初8点开始，每隔5分钟一个计数，输入时间对应的上一个计数值
    
    参数:
        input_time_str: 字符串，格式如 "2025-02-06 04:03"
    
    返回:
        整数，表示最近的上一个5分钟计数的索引
    """
    # 解析输入时间字符串
    # 注意：处理冒号可能是中文全角格式的情况
    input_time_str = input_time_str.replace("：", ":")
    input_time = datetime.strptime(input_time_str, "%Y-%m-%d %H:%M")
    
    # 确定当月月初的日期
    month_start = datetime(input_time.year, input_time.month, 1)
    
    # 设置月初上午8点为起始时间
    start_time = datetime.combine(month_start.date(), time(8, 0))
    
    # 计算从起始时间到输入时间经过了多少分钟
    if input_time < start_time:
        return -1  # 输入时间早于起始时间
        
    delta_minutes = (input_time - start_time).total_seconds() / 60
    
    # 计算经过了多少个完整的5分钟
    kline_idx = int(delta_minutes / 5)
    
    return kline_idx

if __name__ == "__main__":
    print(calculate_kline_idx("2025-02-06 04:03"))
