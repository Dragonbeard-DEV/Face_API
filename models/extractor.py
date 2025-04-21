import numpy as np
import onnxruntime as ort
from PIL import Image

onnx_session = ort.InferenceSession("models/edgeface_fp16.onnx", providers=["CPUExecutionProvider"])

def preprocess(image):
    img = np.asarray(image.resize((112, 112))).astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # Normalize to [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img.astype(np.float32)

def extract_feature(aligned_face):
    input_tensor = preprocess(aligned_face)
    outputs = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input_tensor})
    vec = outputs[0][0]
    vec = vec / np.linalg.norm(vec)
    return vec

