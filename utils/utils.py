import numpy as np
import os
from typing import Callable, Union
import time
import functools
import yaml


def get_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算两个集合中所有边界框的IOU矩阵
    Args:
        boxes1: 形状为[N,4]的numpy数组，每个元素为[x1,y1,x2,y2]
        boxes2: 形状为[M,4]的numpy数组，每个元素为[x1,y1,x2,y2]

    Returns:
        iou_matrix: 形状为[N,M]的numpy数组，其中iou_matrix[i,j]表示boxes1[i]与boxes2[j]的IOU
    """
    if boxes1.shape[0] < 1 or boxes2.shape[0] < 1:
        return np.array([])

    # 扩展维度维度用来适配广播运算
    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    # 计算相交区域的坐标[xx1,yy1,xx2,yy2]
    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    # 计算两个框的面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # 第一个集合的框的面积
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # 第二个集合的框的面积

    # 重叠区域的面积
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    overlap_area = w * h
    union_area = area1 + area2 - overlap_area
    iou_matrix = overlap_area / union_area
    return iou_matrix


def get_class(yaml_file):
    if not os.path.exists(yaml_file):
        raise f"{yaml_file} does not exist!!"
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


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

def x1y1wh2xyxy(x1y1wh:np.ndarray):
    """
    将x1y1wh的bounding box转换成x1y1x2y2的格式
    """
    xyxy = np.copy(x1y1wh)
    xyxy[:,2] = x1y1wh[:,0] + x1y1wh[:,2]
    xyxy[:,3] = x1y1wh[:,1] + x1y1wh[:,3]
    return xyxy


if __name__ =="__main__":
    a = np.array([[1,2,3,4],[5,6,7,8]])
    b = x1y1wh2xyxy(a)
    print(b)

