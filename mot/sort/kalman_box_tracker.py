## 基于https://github.com/roboflow/trackers/blob/main/trackers/core/sort/kalman_box_tracker.py
# 将其修改成中心 + 尺度 + 速度 即[x,y,s,r,vx,vy,vs,vr]的方式，虽然都是8个维度，但是这种方式更加解耦、稳定、易实现
import numpy as np
from numpy.typing import NDArray

class SORTKalmanBoxTracker:
    """
    SortKalmanBoxTracker 是Sort跟踪算法单个跟踪目标的类,并且使用kalman滤波器来预测和更新其位置
    Attributes:
        tracker_id (int): Unique identifier for the tracker.
        number_of_successful_updates (int): Number of times the object has been updated successfully.
        time_since_update (int): Number of frames since the last update. 自上次更新以来的帧数
        state (np.ndarray): State vector of the bounding box. bbox的状态矢量 [x,y,s,r,vx,vy,vs,vr]
        F (np.ndarray): State transition matrix. 状态转移矩阵
        H (np.ndarray): Measurement matrix. 测量矩阵
        Q (np.ndarray): Process noise covariance matrix. 过程噪声协方差矩阵
        R (np.ndarray): Measurement noise covariance matrix. 测量噪声协方差矩阵
        P (np.ndarray): Error covariance matrix. 误差协方差矩阵
        count_id (int): Class variable to assign unique IDs to each tracker. 唯一ID

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].

    """

    count_id: int = 0
    state: NDArray[np.float32]
    F: NDArray[np.float32]
    H: NDArray[np.float32]
    Q: NDArray[np.float32]
    R: NDArray[np.float32]
    P: NDArray[np.float32]

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: NDArray[np.float64],class_id:int) -> None:
        # 初始id设置为 -1 ，当跟踪被认为成熟时将被分配一个正式的id
        self.tracker_id = -1

        # 跟踪的目标的类别id,在传入bbox的同时传入
        self.class_id = class_id

        # 命中次数：表示目标被成功更新的次数
        self.number_of_successful_updates = 1
        # 自上次以来更新的次数
        self.time_since_update = 0

        # (x, y, s, r, vx, vy, vs, vr).
        # x,y: 中心点坐标; s: 面积; r: 宽高比; v: 对应速度
        # We'll store the bounding box in "self.state" 这里需要注意维度 8x1
        self.state = np.zeros((8, 1), dtype=np.float32)

        # 使用第一个检测结果初始化状态self.state
        bbox = self.get_bbox_state(bbox)
        bbox_float: NDArray[np.float32] = bbox.astype(np.float32)
        self.state[0, 0] = bbox_float[0]
        self.state[1, 0] = bbox_float[1]
        self.state[2, 0] = bbox_float[2]
        self.state[3, 0] = bbox_float[3]

        # Basic constant velocity model
        self._initialize_kalman_filter()

    def _initialize_kalman_filter(self) -> None:
        """
        设置卡尔曼滤波器相关的参数
        """
        # State transition matrix (F): 8x8
        # 假设速度恒定
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (H): we directly measure (x, y, s, r)
        self.H = np.eye(4, 8, dtype=np.float32)  # 4x8

        # 过程噪声协方差 (Q)
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # 观测噪声协方差 (R)
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # 误差协方差矩阵 (P)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0

    def predict(self) -> None:
        """
        使用状态转移矩阵,预测bbox的下一个状态
        """
        # 预测状态 x_k = F* x_{k-1}
        self.state = (self.F @ self.state).astype(np.float32)
        # 预测误差协方差 P_k^- = A*P_k*A^T + Q
        self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: NDArray[np.float64]) -> None:
        """
        使用新的更新状态

        Args:
            bbox (np.ndarray): Detected bounding box in the form [x1, y1, x2, y2].
        """
        bbox = self.get_bbox_state(bbox)  # 转换成卡尔曼滤波器处理的状态量
        self.time_since_update = 0
        self.number_of_successful_updates += 1

        # Kalman Gain  np.linalg.inv 方阵的逆矩阵  K_k = P_k*H^T / H*P_k*H^T+R
        S: NDArray[np.float32] = self.H @ self.P @ self.H.T + self.R
        K: NDArray[np.float32] = (self.P @ self.H.T @ np.linalg.inv(S)).astype(np.float32)

        # 残差 z 为测量值 y = z - H*x_k^-
        z: NDArray[np.float32] = bbox.reshape((4, 1)).astype(np.float32)
        y: NDArray[np.float32] = z - self.H @ self.state  # y should be float32 (4,1)

        # Update state
        self.state = (self.state + K @ y).astype(np.float32)

        # Update covariance
        identity_matrix: NDArray[np.float32] = np.eye(8, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

    def get_bbox_state(self, bbox: NDArray[np.float64]) -> NDArray[np.float32]:
        """
        将bbox转换成状态变量的格式

        Args:
            bbox (np.ndarray): Detected bounding box in the form [x1, y1, x2, y2].
        """
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

        return np.array([x, y, s, r])

    def get_state_bbox(self) -> NDArray[np.float32]:
        """
        从状态变量中返回当前的框
        Returns the current bounding box estimate from the state vector.

        Returns:
            np.ndarray: The bounding box [x1, y1, x2, y2]
        """
        state = self.state[:4, 0].flatten().astype(np.float32)
        if state[2] < 0 or state[3]<0: # 当预测的框不符合物理规律时，即出现了负值
            return np.array([0, 0, 0, 0]).astype(int)
        w = np.sqrt(state[2] * state[3])
        h = np.sqrt(state[2] / state[3])

        x1, y1, x2, y2 = state[0] - w / 2, state[1] - h / 2, state[0] + w / 2, state[1] + h / 2

        return np.array([x1, y1, x2, y2]).astype(int)
