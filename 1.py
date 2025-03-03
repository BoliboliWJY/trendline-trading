import numpy as np

def find_first_less_than(arr: np.ndarray, threshold: float) -> int:
    """
    返回第一个小于 threshold 的元素的索引，如果没有符合条件的元素，则返回 -1
    """
    indices = np.where(arr < threshold)[0]
    return indices[0] if indices.size > 0 else -1

# 示例:
if __name__ == "__main__":
    arr = np.array([5, 6, 8, 3, 2])
    threshold = 4
    index = find_first_less_than(arr, threshold)
    print("第一个小于 {} 的元素索引为: {}".format(threshold, index))