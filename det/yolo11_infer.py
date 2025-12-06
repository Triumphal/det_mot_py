#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Date     : 2025/12/5 23:57
@Author   : ArcRay
@FileName : yolo11_infer.py
@Brief    : yolo11 推理 整理
"""

import cv2
import numpy as np
import onnxruntime as ort
from utils.common import get_class
from utils.decorators import cost_time
from typing import List, Tuple
from dataclasses import dataclass
import logging
import os


@dataclass
class Padding:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


@dataclass
class ModelInfo:
    """
    模型信息
    """
    input_name: str  # 输入名称
    output_name: str  # 输出名称
    input_shape: Tuple[int]  # 输入形状 [B,C,H,W]
    output_shape: Tuple[int]  # 输出形状 [B,N,M]


class YOLOv11:
    """
    YOLOv11检测类
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

    def __init__(self, model_path, yaml_file, infer_type="cpu"):
        self.root_path = os.getcwd()
        logging.info(f"root_path: {self.root_path}")
        self.model_path = os.path.join(self.root_path, model_path)  # 模型
        self.yaml_file = os.path.join(self.root_path, yaml_file)  # yaml

        ## ort 日志初始化
        self.session_options = ort.SessionOptions()
        self.session_options.log_severity_level = 3  # 3忽略告警
        self.session = None

        ## 模型信息
        self.model_info = None

        ## 前后处理配置
        self.iou_thr = 0.3
        self.conf_thr = 0.3
        self.padding: Padding = Padding(0, 0, 0, 0)  # (top,bottom,right,left)
        self.ratio = 1  # resize的比率

        ## 原始图像信息
        self.img_width = None
        self.img_height = None

        ## 可视化
        self.classes = get_class(yaml_file)["names"]
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 框颜色

        self.init_model_cost = self.init_onnx_model(infer_type)
        logging.info(f"Init  cost: {self.init_model_cost[-1]:.2f} ms")

    @cost_time()
    def init_onnx_model(self, infer_type="cpu"):
        """
        初始化onnx模型
        Args:
            infer_type : 选择onnxruntime推理使用的device 默认cpu
        """
        providers = ["CPUExecutionProvider"] if infer_type == "cpu" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(self.model_path, self.session_options, providers=providers)
            # 模型的输入
            self.model_info = ModelInfo(
                input_name=self.session.get_inputs()[0].name,
                output_name=self.session.get_outputs()[0].name,
                input_shape=self.session.get_inputs()[0].shape,  # [B,C,H,W]
                output_shape=self.session.get_outputs()[0].shape
            )
        except Exception as e:
            logging.error(f"ONNX model initialization failed, due to {e}")
        logging.info("ONNX model successfully initialized")  # 表示成功加载模型

    @cost_time()
    def preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """
        Returns:
            image_data: (1,3,H,W), 输入读取之后的图像数据
            pad: (top, left)
        """
        img_resize = input_image.copy()
        self.img_height, self.img_width = img_resize.shape[:2]
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        model_input_h, model_input_w = self.model_info.input_shape[2:]
        img_resize = self.letterbox(img_resize, (model_input_h, model_input_w))
        img_resize = np.array(img_resize) / 255.0  # 归一化
        img_resize = np.transpose(img_resize, (2, 0, 1))  # HWC -> CHW
        img_resize = np.expand_dims(img_resize, axis=0).astype(np.float32)  # [C,H,W] -> [B,C,H,W]
        return img_resize

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        输出填充, resize之后的图像和在顶部和左侧填充的多少, 因为是对称所有，输出顶部和左侧
        Args:
            img: 输入的需要resize的图像 [H,W,C]
            new_shape: 目标尺寸大小, [H,W]
        Returns:
            img: resize和padding之后的图像
        """
        img_h, img_w = img.shape[:2]  # current shape [H, W]
        self.ratio = min(new_shape[0] / img_h, new_shape[1] / img_w)  # Scale ratio (new / old)
        new_h, new_w = round(img_h * self.ratio), round(img_w * self.ratio)
        img_resize = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR_EXACT)  # Warning!! 这里的尺寸是 [W,H]

        dh, dw = new_shape[0] - new_h, new_shape[1] - new_w,  # 计算需要填充的值
        # 使用floor(向下)和ceil(向上)取整来保证填充的正确性
        top = int(np.floor(dh / 2))
        bottom = int(np.ceil(dh / 2))  # 剩余填充量给bottom，总和=dh
        left = int(np.floor(dw / 2))
        right = int(np.ceil(dw / 2))  # 剩余填充量给right，总和=dw

        img_resize = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))
        self.padding = Padding(top, bottom, left, right)
        return img_resize

    @cost_time()
    def inference(self, input_data: np.ndarray):
        """
        推理
        """
        model_output = self.session.run(None, {self.model_info.input_name: input_data})
        return model_output

    def set_config(self, conf_thr, iou_thr):
        """
        设置后处理的配置
        Args:
            conf_thr: 置信度阈值
            iou_thr: 后处理使用的iou阈值
        """
        self.conf_thr = conf_thr if conf_thr is not None else self.conf_thr
        self.iou_thr = iou_thr if iou_thr is not None else self.iou_thr

    @cost_time()
    def postprocess(self, model_output: List[np.ndarray], conf_thr: float = None, iou_thr: float = None):
        """
        后处理
        Args:
            model_output: 模型的输出结果 [1, 84, 8400]
            conf_thr : 置信度阈值
            iou_thr: iou阈值
        Return:
            result: 后处理之后的结果，[[[x1, y1, w, h],score,class_id],...]
        """
        self.set_config(conf_thr, iou_thr)  # 设置配置，否者使用默认，conf_thr=0.3  iou_thr=0.3

        outputs = np.transpose(np.squeeze(model_output[0]))  # 移除维度为1的维度并转置 [8400,84]
        bboxes, scores, class_ids = [], [], []

        outputs[:, 0] -= self.padding.left
        outputs[:, 1] -= self.padding.top

        # 迭代输出的结果
        for output in outputs:
            class_scores = output[4:]
            max_score = np.amax(class_scores)

            if max_score >= self.conf_thr:
                class_id = np.argmax(class_scores)
                x, y, w, h = output[0:4]  # 模型输出的结果中心点和宽、高
                # 转换成在原图中的坐标，左上角和宽高格式
                left, top, width, heigh = (
                    int((x - w / 2) / self.ratio),
                    int((y - h / 2) / self.ratio),
                    int(w / self.ratio),
                    int(h / self.ratio),
                )
                scores.append(max_score)
                bboxes.append([left, top, width, heigh])
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(bboxes, scores, self.conf_thr, self.iou_thr)  # bboxes 是左上点和宽高的格式
        result = []
        for i in indices:
            result.append([bboxes[i], scores[i], class_ids[i]])
        return result

    def add_detected_box(self, img, results):
        """
        在img上添加检测框
            results: 后处理的结果 [x1,y1,w,h,score,id]
            save_path: 输出图像保存地址
        """
        for result in results:
            x1, y1, w, h = result[0]
            score = result[1]
            class_id = result[2]
            color = self.color_palette[class_id]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            label = f"{self.classes[class_id]}:{score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x, label_y = int(x1), (int(y1) - 10 if int(y1) - 10 > label_h else int(y1) + 10)
            cv2.rectangle(img, (label_x, label_y - label_h), (label_x + label_w, label_y + label_h), color, cv2.FILLED)
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def infer_one_img(self, img: np.ndarray, conf_thr: float = None, iou_thr: float = None):
        """
        测试单张图片
        Args:
            iou_thr: iou阈值
            conf_thr: 置信度阈值
            img: 输入cv读取的图片的内容 BGR
        Returns:
        """
        result, pre_cost = self.preprocess(img)
        result, infer_cost = self.inference(result)
        result, post_cost = self.postprocess(result, conf_thr=conf_thr, iou_thr=iou_thr)
        cost_times = [pre_cost, infer_cost, post_cost]
        return result, cost_times

## 测试
# if __name__ == "__main__":
#     img_path = "../data/bus.jpg"
#     model_path = "../model/yolo11n.onnx"
#     yaml_path = "../model/coco8.yaml"
#     save_path = "../data/bus_detect.jpg"
#
#     yolo11 = YOLOv11(model_path, yaml_path)
#     detect_result = yolo11.test_img(img_path,save_path)
