import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
from typing import List
import lap
from det.yolo11_infer import YOLOv11
import os
from tqdm import tqdm
from mot.sort.tracker import SORTTracker
import supervision as sv
from utils.utils import x1y1wh2xyxy


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
    tracker = SORTTracker()
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

        # 将结果封装成sv的检测格式
        detections = sv.Detections(
            xyxy= x1y1wh2xyxy(results[:,:4]),
            confidence=results[:,4],
            class_id=results[:,5]
        )

        # 使用SORT进行跟踪 获得跟踪后的结果 [x1,y1,x2,y2,id]
        sv_rets = tracker.update(detections)

        rets = []
        for i in range(len(sv_rets)):
            xyxy = sv_rets[i].xyxy
            trk_id = sv_rets[i].tracker_id
            tmp = np.concatenate([xyxy[0],trk_id]).flatten()
            rets.append([tmp])

        ## 绘制跟踪结果
        for ret in rets:
            try:
                x1, y1, x2, y2, trk_id = map(int, ret[0].tolist())
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 显示跟踪ID
                cv2.putText(frame, f"id:{int(trk_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                continue
        out.write(frame)

    print(f"视频处理完成, 输出文件为{os.path.join(output, 'videos', 'output.mp4')}")


if __name__ == "__main__":
    video_path = "data/pedestrian_1.mp4"
    output = "output"

    
    sort_one_video(video_path, output)
