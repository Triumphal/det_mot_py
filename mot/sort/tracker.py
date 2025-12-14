from typing import List, Set, Sequence
import numpy as np
from mot.sort.kalman_box_tracker import SORTKalmanBoxTracker
from mot.utils import get_iou_matrix
import supervision as sv
from copy import deepcopy
import lap


class SORTTracker:
    """Implements SORT (Simple Online and Realtime Tracking).

    SORT is a pragmatic approach to multiple object tracking with a focus on
    simplicity and speed. It uses a Kalman filter for motion prediction and the
    Hungarian algorithm or simple IOU matching for data association.

    Args:
        lost_track_buffer (int): 当跟踪丢失时需要缓冲的帧数。增加 lost_track_buffer
            值可以增强遮挡处理，显著改善遮挡下的跟踪效果，但可能会增加外观相似的物体发生
             ID 切换的可能性。
        frame_rate (float): 视频帧速率（每秒帧数）。用于计算视频流可以丢失的最大时间。
        track_activation_threshold (float): 用于激活轨迹的检测置信度阈值。只有置信度
            高于此阈值的检测结果才会创建新轨迹，提高此阈值可以减少误报，但可能会遗漏置
            信度较低的真实目标。
        minimum_consecutive_frames (int): 物体必须连续跟踪的帧数，才能被视为“有效”跟踪。
            增加 `minimum_consecutive_frames` 的值可以防止因误检或重复检出而创建意外的
            跟踪器，但可能会遗漏较短的跟踪器。在跟踪器被视为有效之前，其 `tracker_id` 将
            被设置为 `-1`。
        minimum_iou_threshold (float): 将检测结果与现有轨迹关联的 IOU 阈值。
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
    ) -> None:
        # 根据 lost_track_buffer 和 frame_rate 计算无需更新的最大帧数。
        # 这会根据帧速率缩放缓冲区，以确保在不同的帧速率下都能获得一致的基于时间的跟踪。
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold

        # Active trackers
        self.trackers: list[SORTKalmanBoxTracker] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """使用新的检测结果更新跟踪器的状态

        执行卡尔曼滤波器预测，根据 IOU 将检测结果与现有跟踪器关联起来，更新匹配的跟踪器，
        并为未匹配的高置信度检测结果初始化新的跟踪器。

        Args:
            detections (sv.Detections): 当前帧最新的目标检测的结果。

        Returns:
            sv.Detections: A copy of the input detections, augmented with assigned
                `tracker_id` for each successfully tracked object. Detections not
                associated with a track will not have a `tracker_id`.
        """

        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # 对存在的跟踪器预测一个新的位置
        for tracker in self.trackers:
            tracker.predict()

        # 根据检测框和预测框构建IOU成本矩阵
        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        # 从self.state中提取预测框
        predicted_boxes = np.array([t.get_state_bbox() for t in self.trackers])
        iou_matrix = get_iou_matrix(detection_boxes, predicted_boxes)

        # 根据 IOU 将检测框与跟踪器进行匹配
        matched_indices, _, unmatched_detections = self._get_associated_indices(iou_matrix, detection_boxes)

        # 使用指定的检测框更新匹配上的跟踪器
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        # 将没有匹配上的检测框生成新的检测器
        self._spawn_new_trackers(detections, detection_boxes, unmatched_detections)

        # 移除已经"dead"的跟踪器
        self.trackers = self.get_alive_trackers()

        # 获取更新后的检测结果，就是指定了跟踪的id
        updated_detections = self.update_detections_with_track_ids(
            detections,
            detection_boxes,
            self.minimum_iou_threshold,
            self.minimum_consecutive_frames,
        )

        return updated_detections

    def _get_associated_indices(self, iou_matrix: np.ndarray) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        基于IOU的跟踪和检测器关联

        Args:
            iou_matrix (np.ndarray): IOU 成本矩阵.

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices,
                unmatched trackers, unmatched detections.
        """
        detection_nums, tracker_nums = iou_matrix.shape  # 行表示检测框数，列表示预测框数
        matched_indices = []
        unmatched_trackers = set(range(tracker_nums))
        unmatched_detections = set(range(detection_nums))

        if tracker_nums > 0 and detection_nums > 0:
            # 这里修改使用开源的lap.lapjv。
            _, x, y = lap.lapjv(-iou_matrix, extend_cost=True)  # x为每行分配到的列，y为每列分配到的行
            row_indices = list(range(detection_nums))
            col_indices = [x[i] for i in row_indices]
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(col)
                    unmatched_detections.remove(row)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        """
        仅当检测结果的置信度高于阈值时，才创建新的跟踪器。

        Args:
            detections (sv.Detections): The latest set of object detections.
            detection_boxes (np.ndarray): Detected bounding boxes in the form [x1, y1, x2, y2].
        """
        for detection_idx in unmatched_detections:
            if (
                detections.confidence is None
                or detection_idx >= len(detections.confidence)
                or detections.confidence[detection_idx] >= self.track_activation_threshold
            ):
                new_tracker = SORTKalmanBoxTracker(detection_boxes[detection_idx])
                self.trackers.append(new_tracker)

    def get_alive_trackers(self) -> List[SORTKalmanBoxTracker]:
        """
        移除失效或不成熟的丢失轨迹，并获取在 `maximum_frames_without_update` 范围内
        且（已成熟或刚刚更新）的存活轨迹。

        Returns:
            List[SORTKalmanBoxTracker]: List of alive trackers.
        """
        alive_trackers = []
        for tracker in self.trackers:
            is_mature = tracker.number_of_successful_updates >= self.minimum_consecutive_frames
            is_active = tracker.time_since_update == 0
            if tracker.time_since_update < self.maximum_frames_without_update and (is_mature or is_active):
                alive_trackers.append(tracker)
        return alive_trackers
    
    def update_detections_with_track_ids(self, detections: sv.Detections, detection_boxes: np.ndarray) -> sv.Detections:
        """
        该函数会准备带有跟踪 ID 的更新后的检测结果。如果跟踪器“成熟”（>= `minimum_consecutive_frames`）
        或最近更新过，则会为其分配一个 ID，该 ID 指向刚刚更新它的检测结果。

        Args:
            detections (sv.Detections): The latest set of object detections.
            detection_boxes (np.ndarray): Detected bounding boxes in the form [x1, y1, x2, y2].

        Returns:
            sv.Detections: A copy of the detections with `tracker_id` set for each detection that is tracked.
        """
        # Re-run association in the same way (could also store direct mapping)
        # 以相同方式重新运行关联（也可以存储直接映射）
        final_tracker_ids = [-1] * len(detection_boxes)

        # 移除部分跟踪器后，根据当前跟踪器重新计算预测框。
        predicted_boxes = np.array([t.get_state_bbox() for t in self.trackers])
        iou_matrix_final = np.zeros((len(detection_boxes)), len(self.trackers), dtype=np.float32)

        # 在进行第二次 IOU 计算之前，请确保 predicted_boxes 的形状正确。
        if len(predicted_boxes) == 0 and len(self.trackers) > 0:
            predicted_boxes = np.zeros((len(self.trackers), 4), dtype=np.float32)

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            iou_matrix_final = sv.box_iou_batch(predicted_boxes, detection_boxes)

        row_indices, col_indices = np.where(iou_matrix_final > self.minimum_iou_threshold)
        sorted_pairs = sorted(
            zip(row_indices, col_indices),
            key=lambda x: iou_matrix_final[x[0], x[1]],
            reverse=True,
        )
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        for row, col in sorted_pairs:
            # Double check index is in range
            if row < len(self.trackers):
                tracker_obj = self.trackers[int(row)]
                # Only assign if the track is "mature" or is new but has enough hits
                if (int(row) not in used_rows) and (int(col) not in used_cols):
                    if tracker_obj.number_of_successful_updates >= self.minimum_consecutive_frames:
                        # If tracker is mature but still has ID -1, assign a new ID
                        if tracker_obj.tracker_id == -1:
                            tracker_obj.tracker_id = SORTKalmanBoxTracker.get_next_tracker_id()
                        final_tracker_ids[int(col)] = tracker_obj.tracker_id
                    used_rows.add(int(row))
                    used_cols.add(int(col))

        # Assign tracker IDs to the returned Detections
        updated_detections = deepcopy(detections)
        updated_detections.tracker_id = np.array(final_tracker_ids)

        return updated_detections

    def reset(self) -> None:
        """重置追踪器的内部状态

        清除所有处于激活状态的轨迹，并重置轨迹 ID 计数器
        """
        self.trackers = []
        SORTKalmanBoxTracker.count_id = 0
