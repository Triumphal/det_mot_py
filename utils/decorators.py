# 常用的装饰函数
from typing import Callable, Union
import time
import functools


def cost_time() -> Callable:
    """
    计算函数执行时间的装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # 保留被装饰函数的元信息
        def wrapper(*args, **kwargs) -> Union[Callable, tuple]:
            start_time = time.perf_counter()  # 高精度计时器
            result = func(*args, **kwargs)
            return result, (time.perf_counter() - start_time) * 1000

        return wrapper

    return decorator
