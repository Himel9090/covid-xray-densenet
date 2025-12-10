import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="COVID-19 X-ray Classifier (DenseNet121)", layout="centered")
st.title("🩺 COVID–19 | Pneumonia | Normal X-ray Classifier")
st.write("Upload a chest X-ray image and the model will predict the class.")

# -----------------------------
# MODEL PATH
# -----------------------------
MODEL_PATH = "best_densenet121.pth"

# -----------------------------
# TRANSFORM FOR INPUT IMAGES
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# LOAD MODEL FUNCTION
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    # IMPORTANT: weights=None to avoid mismatch error
    model = models.densenet121(weights=None)

    # Replace classifier for 3-class problem
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 3)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model, device

model, device = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = ["COVID", "Normal", "Pneumonia"]

# -----------------------------
# IMAGE PREDICTION FUNCTION
# -----------------------------
def predict(image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)

    return class_names[pred_class.item()], confidence.item()

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing X-ray..."):
            label, confidence = predict(image)

        st.success(f"### 🧠 Model Prediction: **{label}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        # Class-wise probability bars
        st.subheader("Class Probabilities")
        img_tensor = transform(image).unsqueeze(0)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

        for cls, p in zip(class_names, probs):
            st.write(f"{cls}: {p*100:.2f}%")
            st.progress(float(p))
