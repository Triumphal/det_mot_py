import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
from typing import List
import lap
from det.yolo11_infer import YOLOv11
import os
from tqdm import tqdm


class KalmanBoxTracker:
    """
    使用卡尔曼滤波跟踪单个目标的类
    """

    count = 0  # 全局计数器，用于生成唯一的跟踪ID

    def __init__(self, bbox):
        """
        初始化卡尔曼滤波器
        Args:
            bbox : [x1, y1, x2, y2] 格式的检测框
        """
        # 初始化卡尔曼滤波器，状态维度8，观测维度4
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # 状态转移矩阵F_8x8 (x, y, s, r, dx, dy, ds, dr)
        # x,y: 中心点坐标; s: 面积; r: 宽高比; d开头: 对应速度
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # 观测矩阵H_4x8: 只观测(x,y,s,r)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # 过程噪声协方差
        self.kf.Q[4:, 4:] *= 0.01

        # 观测噪声协方差
        self.kf.R[2:, 2:] *= 1.0

        # 初始化状态
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)  # 面积
        r = (x2 - x1) / (y2 - y1)  # 宽高比
        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape(4, 1)

        # 初始化状态估计的协方差矩阵
        self.kf.P[4:, 4:] *= 1000.0  # 速度初始不确定性大
        self.kf.P *= 10.0

        self.time_since_update = 0  # 距离上次更新的帧数
        self.id = KalmanBoxTracker.count  # 跟踪ID
        KalmanBoxTracker.count += 1
        self.history = []  # 历史状态
        self.hits = 0  # 命中次数
        self.hit_streak = 0  # 连续命中次数
        self.age = 0  # 跟踪器存在的帧数

    def update(self, bbox):
        """
        根据检测框更新卡尔曼滤波器状态
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        # 将bbox转换中心点、面积、宽高比格式
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1)
        z = np.array([cx, cy, s, r]).reshape(4, 1)

        # 更新卡尔曼滤波器
        self.kf.update(z)

    def predict(self):
        """
        预测下一帧的目标位置 [x1,y1,x2,y2]
        """
        # 如果速度导致面积为负，重置速度
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        """
        将卡尔曼滤波的状态转换为bbox格式 [x1, y1, x2, y2]
        """
        x = self.kf.x
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        w = np.sqrt(s * r)
        h = s / w
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        return np.array([x1, y1, x2, y2])


class SORT:
    """
    SORT跟踪类
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        初始化SORT跟踪器
        Args:
            max_age: 最大未更新帧数，超过则删除跟踪器
            min_hits: 最小命中次数，达到才输出跟踪结果
            iou_threshold: IOU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []  # 活跃的跟踪器列表
        self.frame_count = 0  # 处理的帧数

    def update(self, dets=np.empty((0, 5))):
        """
        处理一帧的检测结果，返回跟踪结果
        Args:
            dets: 检测框数组，格式为 [[x1,y1,x2,y2,score], ...] 或空数组
        Returns:
            ret: [[x1,y1,x2,y2,track_id], ...]
        """

        self.frame_count += 1
        ret = []
        
        # 1、对所有现有的跟踪器进行预测
        trks = np.zeros((len(self.trackers), 5)) # kf 预测的结果 [x1,y1,x2,y2,id]
        to_del = []
        for i, trk in enumerate(trks):  # 这里修改trk会影响trks,这是由np的特性决定的
            pos = self.trackers[i].predict()  # 预测位置
            trk[:4], trk[4] = pos, self.trackers[i].id
            is_invalid = np.isnan(trk) | np.isinf(trk)
            if np.any(is_invalid):  # 有一个值为无效值放到待删除的列表
                to_del.append(i)
        
        # 删除预测异常的跟踪器 trks要和self.trackers删除的位置一致
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 去除存在无效值的行
        for i in reversed(to_del):  # 反转列表，pop之后不会影响后面的pop的位置
            self.trackers.pop(i)

        # 2、匈牙利算法进行IOU匹配
        matched, unmatched_dets, _ = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 3、更新匹配上的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # 4、为未匹配的检测器创建新跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # 5、收集有效跟踪结果，删除过期跟踪器
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            kf_det = trk.get_state()  # 卡尔曼估计的框

            # 满足命中次数且未过期，输出
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((kf_det, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # 删除过期跟踪器
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret) , trks  # concatenate按行拼接 （跟踪上的结果，预测的结果）

        return np.empty((0, 5)), np.empty((0, 5)) 

    def associate_detections_to_trackers(self, detections: np.array, tracks, iou_threshold):
        """
        将检测结果分配给跟踪对象
        Args:
            detections : 检测的结果 [x1,y1,x2,y2,score]
            tracks     : 上一次的检测结果使用卡尔曼估计的结果 [x1,y1,x2,y2,id]
        Returns:
            matched: 匹配列表 [[i，j],...] detections 和tracks匹配上的位置索引
            unmatched_detections: 没有匹配的检测列表
            unmatched_trackers: 没有匹配的跟踪列表
        """
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # 计算检测框和估计的框的iou
        iou_matrix = self.compute_iou_matrix(detections, tracks)

        # 使用匈牙利算法求解最优匹配，这里使用lapjv是匈牙利算法的优化实现
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:  ## 只有一个iou匹配上的时候直接获得匹配的位置
                matched_indices = np.stack(np.where(a), axis=1)  # 增加一个维度
            else:
                # 匈牙利(Hungarian)算法的指派问题求解，最小权重匹配问题，解决线性指派问题
                _, x, y = lap.lapjv(-iou_matrix, extend_cost=True)  # x为每行分配到的列，y为每列分配到的行
                matched_indices = np.array([[y[i], i] for i in x if i >= 0])
        else:
            matched_indices = np.empty(shape=(0, 2))

        # 筛选iou大于阈值的匹配
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(tracks):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # 过滤已经匹配的低IOU
        matched = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matched.append(m.reshape(1, 2))
        if len(matched) == 0:
            matched = np.empty((0, 2), dtype=int)
        else:
            matched = np.concatenate(matched, axis=0)
        return matched, np.array(unmatched_detections), np.array(unmatched_trackers)

    def compute_iou_matrix(self, boxes1: np.array, boxes2: np.array) -> np.array:
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


def sort_one_video(video_path, output):
    """
    对一个视频进行跟踪
    Params:
        video_path: 需要处理的视频地址
        model_path: 目标检测的模型地址
        output: 带有跟踪的输出视频保存地址
    """
    # 初始化检测模型的参数
    model = "model/yolo11n.onnx"
    yaml_file = "model/coco8.yaml"
    yolo11 = YOLOv11(model, yaml_file)  # 确定模型
    # yolo11.load_config(conf_thr=0.2, iou_thr=0.5)  # 加载配置
    # 初始化跟踪参数
    sort = SORT(max_age=3, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频设置
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.join(output, "videos"), exist_ok=True)
    out = cv2.VideoWriter(
        os.path.join(output, "videos", "output.mp4"), fourcc, fps, (frame_width, frame_height)
    )
    pbr = tqdm(total=frame_counts)
    while True:
        pbr.update(1)
        ret, frame = cap.read()
        if not ret:
            break
        results, _ = yolo11.infer_one_img(frame, conf_thr=0.1, iou_thr=0.2)  # 推理图片

        # 处理检测结果
        class_ids = []
        scores = []
        bboxes = []
        for r in results:
            bboxes.append(r[0])  # [x1,y1,w,h] 左上角 宽 高
            scores.append(r[1])
            class_ids.append(r[2])
        # 组合输入SORT的格式
        dets = []  # [x1,y1,x2,y2]
        for i in range(len(bboxes)):
            x1, y1, w, h = bboxes[i]
            x2, y2 = x1 + w, y1 + h
            dets.append([x1, y1, x2, y2, scores[i]])
        dets = np.array(dets)

        # 使用SORT进行跟踪 获得跟踪后的结果 [x1,y1,x2,y2,id]
        rets, trks = sort.update(dets)

        ## 绘制跟踪结果
        for ret in rets:
            x1, y1, x2, y2, trk_id = map(int, ret.tolist())
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示跟踪ID
            cv2.putText(frame, f"id:{int(trk_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
        # 显示kf预测的结果
        for trk in trks:
            x1, y1, x2, y2, trk_id =map(int, trk.tolist()) 
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            # 显示跟踪ID
            cv2.putText(frame, f"id:{int(trk_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # # 绘制跟踪结果
        # for trk_id, trk_res in tracker.result.items():
        #     x1, y1, x2, y2 = map(int, trk_res[-1].tolist())

        #     # 绘制边界框
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     # 显示跟踪ID
        #     cv2.putText(frame, f"id:{int(trk_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     # 显示轨迹
        #     for trk in trk_res:
        #         cv2.circle(frame, (int(trk[0]), int(trk[1])), 1, (0, 255, 0), 2)
        # 写入输出视频
        out.write(frame)

    print(f"视频处理完成, 输出文件为{os.path.join(output, 'videos', 'output.mp4')}")


if __name__ == "__main__":
    video_path = "data/pedestrian.mp4"
    output = "output"

    
    sort_one_video(video_path, output)
