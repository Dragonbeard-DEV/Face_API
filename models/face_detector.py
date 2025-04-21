import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("models/yolov8s-face-lindevs.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape


def preprocess(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
    img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)
    return input_tensor, img


def detect_face(pil_img):
    input_tensor, orig_img = preprocess(pil_img)
    outputs = session.run(None, {input_name: input_tensor})[0]

    boxes = []
    for det in outputs:
        x1, y1, x2, y2, conf = det[:5]
        if conf > 0.5:
            boxes.append([x1, y1, x2, y2])

    return boxes