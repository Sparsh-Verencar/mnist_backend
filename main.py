# backend/main.py

import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# -------------------------------
# Define your model exactly as in training
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (batch, 32, 7, 7)
        x = x.view(-1, 32 * 7 * 7)             # -> (batch, 1568)
        x = F.relu(self.fc1(x))                # -> (batch, 128)
        x = self.fc2(x)                        # -> (batch, 10)
        return x

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load trained model
# -------------------------------
model = Net()
model.load_state_dict(torch.load("MNIST_SV.pth", map_location="cpu"))
model.eval()  # important for inference

# -------------------------------
# Request schema
# -------------------------------
class ImageData(BaseModel):
    image: str  # base64 string

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(data: ImageData):
    # Remove prefix "data:image/png;base64,"
    image_data = data.image.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Open with PIL, convert to grayscale, resize to 28x28
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28,28))

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Convert to torch tensor and add batch & channel dims
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # shape: [1,1,28,28]

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        pred = int(torch.argmax(output, dim=1).item())

    return {"prediction": pred}
