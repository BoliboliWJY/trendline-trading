import cProfile, pstats, io
import line_profiler


def profile_method(func):
    # how to use: @profile_method before the function you want to profile
    # def wrapper(*args, **kwargs):
    #     profiler = cProfile.Profile()
    #     profiler.enable()
    #     result = func(*args, **kwargs)
    #     profiler.disable()
    #     s = io.StringIO()
    #     sortby = 'tottime'
    #     ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    #     ps.print_stats(10)
    #     print(s.getvalue())
    #     return result
    # return wrapper
    def wrapper(*args, **kwargs):
        profiler = line_profiler.LineProfiler()
        profiler.add_function(func)
        # 执行函数并进行逐行采样
        result = profiler.runcall(func, *args, **kwargs)
        profiler.print_stats()
        return result

    return wrapper
