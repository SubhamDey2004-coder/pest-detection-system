from fastapi import APIRouter, UploadFile, File
from utils.image_processing import process_image
from services.predictor import predict

router = APIRouter()


@router.get("/test")
def test_route():
    return {"message": "Routes working"}


@router.post("/predict")
async def predict_pest(file: UploadFile = File(...)):
    contents = await file.read()
    image = process_image(contents)
    result = predict(image)

    return {
        "filename": file.filename,
        "prediction": result,
    }
