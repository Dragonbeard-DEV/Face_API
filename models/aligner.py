from PIL import Image
from face_alignment.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_alignment.mtcnn_pytorch.src.detector import detect_faces
import numpy as np

REFERENCE = get_reference_facial_points(output_size=(112, 112), inner_padding_factor=0.0, outer_padding=(0, 0), default_square=False)

def align_face(pil_img, bbox):
    img = np.array(pil_img)
    _, landmarks = detect_faces(img)
    if landmarks is not None and len(landmarks) > 0:
        warped = warp_and_crop_face(img, landmarks[0], REFERENCE)
        return Image.fromarray(warped)
    else:
        x1, y1, x2, y2 = map(int, bbox)
        face = pil_img.crop((x1, y1, x2, y2))
        return face.resize((112, 112))