def time_number(interval):
    """ 将时间间隔单位转换为秒数 """
    try:
        time_unit = interval[-1]
        time_value = int(interval[:-1])
    except (ValueError, IndexError):
        raise ValueError("输入格式不正确，应为类似于 '5m', '1h' 等格式")

    # 根据时间单位确定每步的秒数
    if time_unit == 's':  # 秒
        step_seconds = time_value
    elif time_unit == 'm':  # 分钟
        step_seconds = time_value * 60
    elif time_unit == 'h':  # 小时
        step_seconds = time_value * 3600
    elif time_unit == 'd':  # 天
        step_seconds = time_value * 86400
    elif time_unit == 'w':  # 周
        step_seconds = time_value * 604800
    else:
        raise ValueError("不支持的时间单位")
    return step_seconds
