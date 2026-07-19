from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.pipeline.process_video import process_video

app = FastAPI()

TEMP_FOLDER="temp"

os.makedirs(TEMP_FOLDER,exist_ok=True)


@app.get("/health")

def health():

    return {"status":"ok"}



@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    video_path=os.path.join(
        TEMP_FOLDER,
        file.filename
    )

    with open(video_path,"wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    results=process_video(video_path)

    return {
        "predictions":results
    }