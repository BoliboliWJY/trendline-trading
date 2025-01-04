import cProfile

# 定义要分析的函数
def example_function_1():
    total = 0
    for i in range(1000000):
        total += i
    return total

def example_function_2():
    total = 0
    for i in range(500000):
        total += i
    return total

# 创建 profiler 对象
profiler = cProfile.Profile()

# 启用性能分析
profiler.enable()

# 执行要分析的函数
example_function_1()
example_function_2()

# 停止性能分析
profiler.disable()

# 打印分析结果
profiler.print_stats(sort='cumtime')  # 'cumtime' 按照累计时间排序，可以改为 'time' 按照单次调用时间排序
