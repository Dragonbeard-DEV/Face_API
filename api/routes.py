from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

from models.face_detector import detect_face
from models.aligner import align_face
from models.extractor import extract_feature
from models.database import add_to_faiss, search_in_faiss

router = APIRouter()

@router.post("/enroll")
async def enroll(image: UploadFile = File(...), name: str = Form(...), group: str = Form("default")):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        bboxes = detect_face(img)
        if not bboxes:
            return JSONResponse(status_code=404, content={"error": "No face detected"})

        aligned = align_face(img, bboxes[0])
        feature = extract_feature(aligned)
        add_to_faiss(feature, name, group)

        return {"status": "success", "name": name, "group": group, "feature_dim": len(feature)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/recognize")
async def recognize(image: UploadFile = File(...), group: str = Form("default")):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        bboxes = detect_face(img)
        if not bboxes:
            return JSONResponse(status_code=404, content={"error": "No face detected"})

        aligned = align_face(img, bboxes[0])
        feature = extract_feature(aligned)

        name, similarity = search_in_faiss(feature, group)
        matched = similarity >= 0.7

        return {"matched": matched, "name": name, "similarity": float(similarity), "group": group}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

