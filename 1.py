import numpy as np
def flatten_to_array(trend_high):
    """
    先过滤掉空的子数组，然后利用 np.concatenate 将非空子数组拼接成一个一维数组。
    假设 trend_high 中的每个非空子数组都是 NumPy 数组。
    """
    filtered = [sub for sub in trend_high if sub.size > 0]
    if filtered:
        return np.concatenate(filtered)
    else:
        return np.array([])
if __name__ == "__main__":
    # 示例：构造一些数据。这里假设每个子数组都是 NumPy 数组。
    trend_high = []
    import random
    for _ in range(30000):
        if random.random() < (5000 / 30000):
            # 生成随机长度为1到3的 NumPy 数组
            sub = np.arange(random.randint(1, 3))
        else:
            sub = np.array([])
        trend_high.append(sub)
    import time
    start = time.time()
    flat_array = flatten_to_array(trend_high)
    end = time.time()
    print("扁平化后 NumPy 数组长度（约）：", flat_array.shape[0])
    print("处理耗时: {:.6f} 秒".format(end - start))