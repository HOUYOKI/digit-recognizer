# 🔢 Digit Recognizer App (Wetaan Task)

A simple deep learning project that recognizes hand-drawn digits (0–9) using a trained PyTorch model.  
Built using a Streamlit UI for real-time digit prediction.

---

## 📁 Project Structure
digit-recognizer/
├── DigitRecApp/ ← Streamlit web app
│ ├── app.py
├── Wetaan_Task_Digit_Recogniser.ipynb ← Model training notebook (Google Colab)
├── digit_model.pth ← Trained model file (PyTorch)
└── README.md 

---

## 🧠 Model Training (Google Colab)

The model was trained on the MNIST dataset using a CNN in PyTorch.  
To view or reproduce the training process:

1. Open `Wetaan_Task_Digit_Recogniser.ipynb`  
2. Run in Google Colab  
3. It will output `digit_model.pth`, which is used in the app.

---

## 🖥️ Run the App

### 🔧 Install dependencies

Make sure you have Python and Streamlit installed.  
Inside the `DigitRecApp/` folder, create a `requirements.txt` like this:

```txt
streamlit
torch
torchvision
numpy

