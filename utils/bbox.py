# !/usr/bin/env python3

# @Time    : 2025/07/06 16:28
# @Author  : ArcRay
# @FileName: bbox.py
# @Brief   : bbox 相关的操作公共函数
import numpy as np
from typing import List, Tuple
import time


def compute_iou_matrix(boxes1: np.array, boxes2: np.array):
    """
    计算两个集合中所有边界框的IOU矩阵
    Args:
        boxes1: 形状为[N,4]的numpy数组，每个元素为[x1,y1,x2,y2]
        boxes2: 形状为[M,4]的numpy数组，每个元素为[x1,y1,x2,y2]

    Returns:
        iou_matrix: 形状为[N,M]的numpy数组，其中iou_matrix[i,j]表示boxes1[i]与boxes2[j]的IOU
    """

    # 扩展维度维度用来适配广播运算
    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    # 计算相交区域的坐标[xx1,yy1,xx2,yy2]
    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    overlap_area = w * h
    union_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1]) + \
                 (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) - \
                 overlap_area
    iou_matrix = overlap_area / union_area
    return iou_matrix


# 测试代码
if __name__ == "__main__":
    # 创建两个包含多个边界框的集合
    boxes_a = np.array([
        [1, 1, 4, 3],  # 框1
        [2, 2, 5, 5],  # 框2
        [0, 0, 2, 2]  # 框3
    ], dtype=np.float32)

    boxes_b = np.array([
        [2, 2, 4, 4],  # 框A
        [1, 1, 3, 3]  # 框B
    ], dtype=np.float32)

    # 计算IoU矩阵
    start_time = time.perf_counter()
    iou_matrix1 = compute_iou_matrix(boxes_a, boxes_b)
    # iou_matrix2 = iou_batch(boxes_a, boxes_b)
    print(f"compute_iou_matrix1: {time.perf_counter() - start_time:.7f}")
