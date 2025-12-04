# 常用的装饰函数
from typing import Callable, Union
import time
import functools


def get_cost_timer(print_time: bool = False, return_time: bool = True) -> Callable:
    """
    计算函数执行时间的装饰器
    
    参数:
        print_time: 是否打印执行时间 默认False
        return_time: 是否将执行时间作为额外返回值 默认True
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # 保留被装饰函数的元信息
        def wrapper(*args, **kwargs) -> Union[Callable, tuple]:
            start_time = time.perf_counter()  # 高精度计时器
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            # 打印执行时间
            if print_time:
                print(f"函数 {func.__name__} 执行时间: {execution_time:.6f} s")
            # 返回执行时间
            if return_time:
                return result, execution_time
            else:
                return result

        return wrapper

    return decorator
