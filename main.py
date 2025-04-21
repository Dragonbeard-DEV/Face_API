from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from api.routes import router
import uvicorn

app = FastAPI(title="Face Recognition API")
app.include_router(router)

# Để chạy public trên server thật, bỏ reload=True khi deploy
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
