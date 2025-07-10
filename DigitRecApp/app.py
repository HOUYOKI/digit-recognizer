import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np

# ─────────────────────────────
# Helper: Crop and Center Drawing
# ─────────────────────────────
def crop_and_center(img):
    img_np = np.array(img)

    coords = np.column_stack(np.where(img_np > 0))
    if coords.size == 0:
        return img.resize((28, 28))  # fallback

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_np[y_min:y_max+1, x_min:x_max+1]
    cropped_img = Image.fromarray(cropped).resize((20, 20))
    new_img = Image.new("L", (28, 28), color=0)
    new_img.paste(cropped_img, (4, 4))  # center it

    return new_img

# ─────────────────────────────
# Model Definition
# ─────────────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ─────────────────────────────
# Load Model
# ─────────────────────────────
model = DigitCNN()
model.load_state_dict(torch.load("digit_model_v2.pth", map_location=torch.device('cpu')))
model.eval()

# ─────────────────────────────
# Streamlit UI
# ─────────────────────────────
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✏️ Draw a Digit")
st.write("Draw a number between 0–9 and click **Predict**.")

canvas_result = st_canvas(
    fill_color="rgba(255,255,255,1)",  # White canvas
    stroke_width=12,
    stroke_color="rgba(0,0,0,1)",      # Black ink
    background_color="rgba(255,255,255,1)",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ─────────────────────────────
# Prediction
# ─────────────────────────────
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0].astype('uint8')
        img = Image.fromarray(img, mode='L')
        img = ImageOps.invert(img)
        img = crop_and_center(img)

        # Show model input preview
        st.image(img, caption="🧠 What the model sees", width=150)

        img_tensor = transforms.ToTensor()(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        st.success(f"✅ Predicted Digit: **{prediction}**")
    else:
        st.warning("✏️ Please draw something first!")