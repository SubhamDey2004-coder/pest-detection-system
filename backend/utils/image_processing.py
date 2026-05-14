from PIL import Image
import io

def process_image(image_bytes: bytes):
    # Convert bytes -> PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize image (standard size for models)
    image = image.resize((224, 224))
    
    return image

