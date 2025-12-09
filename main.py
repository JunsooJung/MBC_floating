import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin")

import supervision as sv
from inference import get_roboflow_model
import numpy as np
import cv2
from tqdm import tqdm

try:
    import onnxruntime as ort
    print("ONNXRuntime providers:", ort.get_available_providers())
except Exception as e:
    print("ONNXRuntime import error:", e)

MODEL_ID = "[modelid]"
API_KEY = "[apikey]"
VIDEO_PATH = "[saveroot]"
OUTPUT_VIDEO_PATH = ("[resultroot]")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print("총 프레임 수:", frame_count)

progress = tqdm(total=frame_count, desc="Processing")
model = get_roboflow_model(model_id=MODEL_ID, api_key=API_KEY)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    progress.update(1)
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    annotated_image = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene = annotated_image, detections=detections)
    return annotated_image

# Process the video
sv.process_video(
    source_path=VIDEO_PATH,
    target_path=OUTPUT_VIDEO_PATH,
    callback=callback
)

progress.close()
print("처리 완료")
