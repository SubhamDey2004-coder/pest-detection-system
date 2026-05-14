from fastapi import APIRouter, UploadFile, File
from utils.image_processing import process_image
from services.predictor import predict

router = APIRouter()

@router.get("/test")
def test_route():
    return {"message": "Routes working"}

@router.post("/predict")
async def predict_pest(file: UploadFile = File(...)):
    # Step 1: Read image file
    contents = await file.read()
    # Step 2: Process image
    image = process_image(contents)

    # Step 3: Dummy prediction (temporary)
    result = predict(image)

    return {
        "filename": file.filename,
        "prediction": result
    }