import supervision as sv
from mot.sort.tracker import SORTTracker
from det.yolo11_infer import YOLOv11
from utils.utils import x1y1wh2xyxy
import numpy as np
from tqdm import tqdm


def mot_test_with_sv(model: YOLOv11, source_path: str, target_path: str):
    """
    使用 supervision 对 mot 进行测试
    """
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = SORTTracker()

    def mot_callback(frame: np.ndarray, index: int):
        results, _ = model.infer_one_img(frame, conf_thr=0.1, iou_thr=0.2)
        detections = model.get_sv_result(results)
        detections = tracker.update(detections)
        labels = [
                f"id:{tracker_id} {model.classes_name[class_id]}"
                for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
            ]
        # 标注
        annotated_frame = box_annotator.annotate(frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
        return annotated_frame

    sv.process_video(
        source_path=source_path, target_path=target_path, callback=mot_callback, show_progress=True
    )


def mot_test_with_sv_1(model: YOLOv11, source_path: str, target_path: str):
    video_info = sv.VideoInfo.from_video_path(source_path)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = SORTTracker()
    frame_generator = sv.get_video_frames_generator(source_path)
    with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results, _ = model.infer_one_img(frame, conf_thr=0.1, iou_thr=0.2)
            detections = model.get_sv_result(results)
            detections = tracker.update(detections)
            labels = [
                f"id:{tracker_id} {model.classes_name[class_id]}"
                for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
            ]
            annotated_frame = box_annotator.annotate(frame, detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    model = "model/yolo11n.onnx"
    yaml_file = "model/coco8.yaml"
    model = YOLOv11(model, yaml_file)

    source_path = "data/pedestrian_1.mp4"
    target_path = "output/pedestrian_1_mot.mp4"
    mot_test_with_sv(model, source_path, target_path)
