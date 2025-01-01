# Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import timm

# Initialize FastAPI app
app = FastAPI()

# Define the transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
num_classes = 24  # This should be the same as used during training
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('swin_skin_disease_model_with3.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the prediction function
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# API endpoint for predictions
@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image = Image.open(file.file).convert('RGB')
        predicted_class = predict_image(image)
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
