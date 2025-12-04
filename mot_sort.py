# SORT 跟踪算法
from filterpy.kalman import KalmanFilter
import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment
import cv2
from yolo11_onnx import YOLOv11
import os
import lap
from utils.bbox import compute_iou_matrix
from tqdm import tqdm


def convert_bbox_to_x(bbox):
    """
    将边界框转换为观测向量 x 
        bbox: [x, y, w, h] (中心坐标和宽高)
        返回: [x, y, s, r] (观测向量)
    """
    x, y = bbox[0], bbox[1]
    s = bbox[2] * bbox[3]  # s, 计算面积
    r = bbox[2] / bbox[3]  # r, 宽高比
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    将中心形式的边界框[x,y,s,r]转换成[x,y,w,h]的形式，主要用于KalmanFilter
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0], x[1], w, h]).reshape((1, 4))


# Kalman滤波估计
class KalmanBoxTracker:
    """
    跟踪单个目标的状态
    """
    count = 0

    def __init__(self, bbox):
        """
        使用初始边界框初始化跟踪器
        """
        self.kf = None
        self.kf_param_initial(bbox)  # 初始化kf参数
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def kf_param_initial(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # 设置状态转移矩阵F x_k =  F * x_{k-1} + w_{k-1} w~N(0, Q)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]])
        # 设置过程噪声协方差矩阵Q
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # 设置观测矩阵H y_k = H * x_k + v_k v~N(0, R)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]])
        # 设置观测噪声协方差矩阵R
        self.kf.R[2:, 2:] *= 10.0
        # 后验估计的状态估计误差的协方差矩阵P
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        # 设置初始状态 x_k = [x, y, s, r, vx=0, vy=0, vr=0]
        self.kf.x[:4] = convert_bbox_to_x(bbox)

    def update(self, bbox):
        """
        使用观测到的边界框更新跟踪器状态
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_x(bbox))

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计 测量更新
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将检测结果分配给跟踪对象
        Returns:
            matched: 匹配列表
            unmatched_detections: 没有匹配的检测列表
            unmatched_trackers: 没有匹配的跟踪列表
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = compute_iou_matrix(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 匈牙利(Hungarian)算法的指派问题求解，最小权重匹配问题，解决线性指派问题
            _, x, y = lap.lapjv(-iou_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
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


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        设置SORT的关键参数
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.result = dict()  # 记录目标id的中心点的历史位置{id:[[x,y,w,h],...]}
        np.random.seed(0)

    def update(self, detections=np.empty((0, 5))):
        """
        更新每一帧的结果，即使该帧检测为空也需运行该方法，更新result
        Params:
            detections: 每个检测框坐标形式为 左上角点和右下角点[x1,y1,x2,y2,score]
        """
        self.frame_count += 1
        # 从已经有的轨迹中预测位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for i, trk in enumerate(trks):
            pos = self.trackers[i].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # 有一个值为空就放到待删除的列表
                to_del.append(i)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 去除存在无效值的行
        for i in reversed(to_del):  # 反转列表，pop之后不会影响后面的pop
            self.trackers.pop(i)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trks, self.iou_threshold)

        # 根据分配之后的检测更新匹配的跟踪器
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :])

        # 对于未匹配的检测到的detections初始化新的跟踪器
        for i in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(new_trk)

        # 根据跟踪之后的结果，输出当前的结果
        i = len(self.trackers)
        temp_result = dict()  # 临时记录结果
        for trk in reversed(self.trackers):
            box = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                if trk.id in self.result.keys():
                    temp_result.update({trk.id: self.result[trk.id]})
                    temp_result[trk.id].append(box)
                else:
                    temp_result.update({trk.id: [box]})
            i -= 1
            # 移除已经消失的轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        self.result = temp_result


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
    yaml_file = "../model/coco8.yaml"
    yolo11 = YOLOv11(model, yaml_file)  # 确定模型
    yolo11.load_config(conf_thr=0.2, iou_thr=0.5)  # 加载配置
    # 初始化跟踪参数
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
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
    os.makedirs(os.path.join(output, 'videos'), exist_ok=True)
    out = cv2.VideoWriter(os.path.join(output, "videos", 'output.mp4'), fourcc, fps, (frame_width, frame_height))

    for i in tqdm(range(frame_counts)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        results, _ = yolo11.inference(input_image=frame, inference_type="cuda")
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

        # 使用SORT进行跟踪
        tracker.update(dets)
        # 绘制跟踪结果
        for trk_id, trk_res in tracker.result.items():
            x1, y1, x2, y2 = map(int, trk_res[-1].tolist())

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示跟踪ID
            cv2.putText(frame, f"id:{int(trk_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 显示轨迹
            for trk in trk_res:
                cv2.circle(frame, (int(trk[0]), int(trk[1])), 1, (0, 255, 0), 2)
        # 写入输出视频
        out.write(frame)

    print(f"视频处理完成, 输出文件为{os.path.join(output, 'videos', 'output.mp4')}")


if __name__ == "__main__":
    sort_one_video("../data/PETS09-S2L1-raw.webm", "output")
