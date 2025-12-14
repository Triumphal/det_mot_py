import numpy as np



def get_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算两个集合中所有边界框的IOU矩阵
    Args:
        boxes1: 形状为[N,4]的numpy数组，每个元素为[x1,y1,x2,y2]
        boxes2: 形状为[M,4]的numpy数组，每个元素为[x1,y1,x2,y2]

    Returns:
        iou_matrix: 形状为[N,M]的numpy数组，其中iou_matrix[i,j]表示boxes1[i]与boxes2[j]的IOU
    """
    if boxes1.shape < 1 or boxes2.shape <1:
        return np.empty([])

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