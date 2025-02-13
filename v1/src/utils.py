import cProfile, pstats, io

def profile_method(func):
    # how to use: @profile_method before the function you want to profile
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(10)
        print(s.getvalue())
        return result
    return wrapper 