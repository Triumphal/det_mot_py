# yolo检测

import onnxruntime as ort
import cv2
import yaml
import os
import numpy as np
from typing import List, Tuple
import time
from utils.decorators import get_cost_timer
from pathlib import Path
from tqdm import tqdm


def get_class(yaml_file):
    if not os.path.exists(yaml_file):
        raise f"{yaml_file} does not exist!!"
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def letterbox(
    img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    输出填充, resize之后的图像和在顶部和左侧填充的多少, 因为是对称所有，输出顶部和左侧
    Args:
        img: 输入的需要resize的图像 HWC
        new_shape: 目标尺寸大小, (H,W)
    Returns:
        img: resize和padding之后的图像
        pad: padding的形状(top,left)
    """
    shape = img.shape[:2]  # # current shape [height, width]
    ratio = min(
        new_shape[0] / shape[0], new_shape[1] / shape[1]
    )  # Scale ratio (new / old)
    new_unpad = int(round(shape[1] * ratio)), int(
        round(shape[0] * ratio)
    )  # computing padding [w, h]
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    pad = (top, left)
    return img, pad


class YOLOv11:
    """
    YOLOv11检测类
    """

    def __init__(self, model_path, yaml_file):
        """
        初始化,加载模型和输出的类别
        """
        self.img_width = None
        self.img_height = None
        self.iou_thr = 0.3
        self.conf_thr = 0.3
        self.input_name = None
        self.input_height = None
        self.input_width = None
        self.model_inputs = None
        self.session = None
        self.model_path = model_path  # 模型
        self.is_load_onnx = False  # 用来判断onnx是否已经加载
        self.cost_time: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # 模块耗时,[模型预加载耗时(模型已加载显示的是初始的耗时)，预处理耗时，推理耗时，后处理耗时]
        self.classes = get_class(yaml_file)["names"]  # 类别
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3)
        )  # 框颜色

    @get_cost_timer()
    def load_onnx_model(self, inference_type="cpu") -> bool:
        """
        使用ort加载模型onnx模型
        """
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # 3忽略告警
        if inference_type == "cpu":
            providers = ["CPUExecutionProvider"]
        elif inference_type == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise 'please select inference_type in ["cpu","cuda"]'
        try:
            self.session = ort.InferenceSession(
                self.model_path, session_options, providers=providers
            )
            # 得到模型的输入
            self.model_inputs = self.session.get_inputs()
            self.input_width = self.model_inputs[0].shape[2]
            self.input_height = self.model_inputs[0].shape[3]
            self.input_name = self.model_inputs[0].name
        except Exception as e:
            print(e)
            return False
        return True  # 表示成功加载模型

    def load_config(self, conf_thr, iou_thr):
        """
        加载配置
        Args:
            conf_thr: 置信度阈值
            iou_thr: 后处理使用的iou阈值
        """
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    @get_cost_timer()
    def preprocess(self, input_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Returns:
            image_data: (1,3,H,W), 输入读取之后的图像数据
            pad: (top, left)
        """
        img = input_image.copy()
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img, pad = letterbox(img, (self.input_width, self.input_height))  # 填充图像 HWC
        image_data = np.array(img) / 255.0  # 归一化
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC ->CHW
        image_data = np.expand_dims(image_data, axis=0).astype(
            np.float32
        )  # CHW -> NCHW
        return image_data, pad

    @get_cost_timer()
    def postprocess(self, model_output: List[np.ndarray], pad: Tuple[int, int]):
        """
        后处理
        Args:
            model_output: 模型的输出结果
            pad: 预处理时图片的padding的尺寸(top, left)
        Return:
            result: 后处理之后的结果，[[[x1, y1, w, h],score,class_id],...]

        """
        outputs = np.transpose(np.squeeze(model_output[0]))

        bboxes, scores, class_ids = [], [], []

        # 计算边界框的缩放因子
        gain = min(
            self.input_height / self.img_height, self.input_width / self.img_width
        )
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # 迭代输出的结果
        for output in outputs:
            class_scores = output[4:]
            max_score = np.amax(class_scores)

            if max_score >= self.conf_thr:
                class_id = np.argmax(class_scores)
                x, y, w, h = (
                    output[0],
                    output[1],
                    output[2],
                    output[3],
                )  # 模型输出的结果中心点和宽、高
                # 转换成在原图中的坐标，左上角和宽高格式
                left, top, width, heigh = (
                    int((x - w / 2) / gain),
                    int((y - h / 2) / gain),
                    int(w / gain),
                    int(h / gain),
                )
                scores.append(max_score)
                bboxes.append([left, top, width, heigh])
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            bboxes, scores, self.conf_thr, self.iou_thr
        )  # bboxes 是左上点和宽高的格式
        result = []
        for i in indices:
            result.append([bboxes[i], scores[i], class_ids[i]])
        return result

    def inference(self, input_image: np.ndarray, inference_type="cpu"):
        """
        推理
        """
        # 模型预加载
        if not self.is_load_onnx:
            self.is_load_onnx, self.cost_time[0] = self.load_onnx_model(
                inference_type=inference_type
            )

        # 数据预处理
        (img_data, pad), self.cost_time[1] = self.preprocess(input_image)

        # 推理
        start_time = time.perf_counter()
        model_output = self.session.run(None, {self.input_name: img_data})
        self.cost_time[2] = time.perf_counter() - start_time

        # 后处理
        postprocess_results, self.cost_time[3] = self.postprocess(model_output, pad)

        return postprocess_results, self.cost_time

    def add_detected_box(self, img, postprocess_results):
        """
        添加检测框
            img: 输入图片
            postprocess_results: 后处理的结果 [x1y1wh,score,id]
            save_path: 输出图像保存地址
        """
        for result in postprocess_results:
            x1, y1, w, h = result[0]
            score = result[1]
            class_id = result[2]
            color = self.color_palette[class_id]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            label = f"{self.classes[class_id]}:{score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_x, label_y = int(x1), (
                int(y1) - 10 if int(y1) - 10 > label_h else int(y1) + 10
            )
            cv2.rectangle(
                img,
                (label_x, label_y - label_h),
                (label_x + label_w, label_y + label_h),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return img

    def test_one_image(self, image_path):
        ori_img = cv2.imread(image_path)
        image_name = Path(image_path).name
        self.load_config(conf_thr=0.2, iou_thr=0.5)  # 加载配置
        results, cost_time = self.inference(input_image=ori_img, inference_type="cpu")  # 推理图片
        img_boxes = self.add_detected_box(ori_img, results)
        print(f"cost time")
        print(f"load mode  : {cost_time[0]:.4f} s ")
        print(f"preprocess : {cost_time[1]:.4f} s ")
        print(f"inference  : {cost_time[2]:.4f} s ")
        print(f"postprocess: {cost_time[3]:.4f} s ")
        print(f"total      : {sum(cost_time):.4f} s")
        save_root_path = "../output/images/"
        os.makedirs(save_root_path,exist_ok=True)
        cv2.imwrite(os.path.join(save_root_path,image_name), img_boxes)

    def test_one_video(self, video_path):
        video_name = Path(video_path).name
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() is False:
            print("Error opening video stream or file")
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"video fps:{fps},width:{frame_width},height:{frame_height} frame_counts:{frame_counts}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_root_path = "../output/images/"
        os.makedirs(save_root_path,exist_ok=True)
        out = cv2.VideoWriter(os.path.join(save_root_path,video_name),fourcc,fps,(frame_width, frame_height))

        # 初始化tqdm进度条
        pbar = tqdm(total=frame_counts, desc="Processing Frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results, cost_time = self.inference(input_image=frame, inference_type="cpu")  # 推理图片
            frame_boxes = self.add_detected_box(frame, results)
            out.write(frame_boxes)
            pbar.update(1)  # 更新进度条



if __name__ == "__main__":
    model = "../model/yolo11n.onnx"
    yaml_file = "../model/coco8.yaml"
    yolo11 = YOLOv11(model, yaml_file)  # 确定模型

    # 测试图片
    # image_path = "../data/bus.jpg"
    # print(f"测试图片,{image_path}")
    # yolo11.test_one_image(image_path)

    # 测试视频
    video_path = "../data/行人检测测试视频_1080p.mp4"
    print(f"测试视频,{video_path}")
    yolo11.test_one_video(video_path)
