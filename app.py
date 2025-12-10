import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os

# ---------------------------
# 1. Download model from Google Drive
# ---------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1qRJINBpLEEwUkOxz7oNhoXYMcQnSeUHW"
MODEL_PATH = "best_densenet121.pth"

@st.cache_resource
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    # Load DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)  # 3 classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# ---------------------------
# 2. Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["COVID", "Normal", "Pneumonia"]

# ---------------------------
# 3. Streamlit UI
# ---------------------------
st.title("🩺 Chest X-Ray Classifier — DenseNet121")
st.write("Upload a chest X-ray image to classify COVID, Normal, or Pneumonia.")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = classes[pred.item()]

    st.success(f"### 🩻 Prediction: **{label}**")
