#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Date     : 2025/12/6 18:08 
@Author   : ArcRay 
@FileName : det_test.py
@Brief    : 检测测试
"""

from det.yolo11_infer import YOLOv11
import cv2
import logging
from pathlib import Path
import os
from tqdm import tqdm


def test_img(model: YOLOv11, image_path, save_path):
    """
    测试单张图片
    Args:
        image_path: 输入的图片的地址
    Returns:
    """
    img = cv2.imread(image_path)
    result, cost_times = model.infer_one_img(img)  # 推理一张图包含（前处理，模型推理，后处理）
    ## 保存
    img_with_bbox = model.add_detected_box(img, result)
    cv2.imwrite(save_path, img_with_bbox)

    logging.info(f"pre   cost: {cost_times[0]:.2f} ms")
    logging.info(f"infer cost: {cost_times[1]:.2f} ms")
    logging.info(f"post  cost: {cost_times[2]:.2f} ms")


def test_video(model: YOLOv11, video_path, save_path):
    video_name = Path(video_path).name
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video fps:{fps},width:{frame_width},height:{frame_height} frame_counts:{frame_counts}")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # MP4格式的H.264
    os.makedirs(Path(save_path).parent, exist_ok=True)
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    # 初始化tqdm进度条
    pbar = tqdm(total=frame_counts, desc="Processing Frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results, cost_time = model.infer_one_img(frame, conf_thr=0.1, iou_thr=0.2)  # 推理图片
        frame_boxes = model.add_detected_box(frame, results)
        out.write(frame_boxes)
        pbar.update(1)  # 更新进度条


if __name__ == '__main__':
    model_path = "model/yolo11n.onnx"
    yaml_path = "model/coco8.yaml"
    yolo11 = YOLOv11(model_path, yaml_path)

    # # 测试图片
    # img_path = "data/bus.jpg"
    # save_img_path = "output/bus_detect.jpg"
    # test_img(yolo11,img_path, save_img_path)

    # 测试视频
    video_path = "data/pedestrian_1.mp4"
    save_video_path = "output/pedestrian_1_detect.mp4"
    test_video(yolo11, video_path, save_video_path)
