<img width="441" height="148" alt="Screenshot 2026-05-08 161707" src="https://github.com/user-attachments/assets/b5d06b82-8171-4350-ab23-873a7a008cb6" /># 🔬 DermAI: Skin Cancer Triage Assistant

An AI-powered dermoscopic image classification system that detects **benign vs malignant** skin lesions using CNN deep learning, with an intelligent triage note generator powered by the **Cohere API**.

Built as an Open Ended Lab project for AI & Machine Learning course at IIUI.

---

## 🖥️ Demo


<img width="832" height="362" alt="Screenshot 2026-06-06 140728" src="https://github.com/user-attachments/assets/ee3da340-a027-4e6b-ad93-5af3c1261b83" />

<img width="620" height="335" alt="Screenshot 2026-06-06 140650" src="https://github.com/user-attachments/assets/94e326c0-b131-48ea-965e-d3840ac7294f" />

<img width="749" height="307" alt="Screenshot 2026-06-06 140749" src="https://github.com/user-attachments/assets/3321eb93-93c2-4d61-9ce6-dd735986fa17" />



---

## ✨ Features

- 🧠 CNN model trained on ISIC Skin Cancer Dataset (Kaggle)
- ✅ Classifies skin lesions as **Benign** or **Malignant**
- 📋 Auto-generates triage notes using **Cohere API**
- 🌐 Beautiful **Gradio** web interface
- 🔒 Secure API key management via python-dotenv
- ⚠️ Error handling for corrupted images and unsupported formats
- 📊 Shows confidence score for each prediction

---

## 🏗️ CNN Architecture

```
Input (224x224x3)
    ↓
Conv2D + ReLU + MaxPooling  (Feature extraction)
    ↓
Conv2D + ReLU + MaxPooling  (Deep features)
    ↓
Flatten → Dense(128) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(1) + Sigmoid → [Benign / Malignant]
```

---

## 📁 Project Structure

```
skin-cancer-triage/
├── final.ipynb              ← Training + full pipeline notebook
├── skin_cancer_app.py       ← Gradio app with Cohere integration
├── skin_cancer_model.h5     ← Trained CNN model
├── .env                     ← API keys (not uploaded)
├── train/                   ← Training images (not uploaded)
├── test/                    ← Test images (not uploaded)
└── train_concat.csv         ← Dataset metadata (not uploaded)
```

---

## 📦 Dataset

This project uses the **ISIC Skin Cancer Dataset** from Kaggle.

**Download here:**
https://www.kaggle.com/datasets/nroman/melanoma-external-malignant-256

After downloading:
1. Extract training images to `train/` folder
2. Extract test images to `test/` folder
3. Place `train_concat.csv` in the root folder

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```
pip install tensorflow opencv-python gradio cohere python-dotenv numpy pandas
```

### 2. Set up API key

Create a `.env` file in the project folder:
```
COHERE_API_KEY=your_cohere_api_key_here
```

Get your free API key at: https://cohere.com

### 3. Download dataset

Follow the Dataset section above.

### 4. Train the model

Run `final.ipynb` in Jupyter Notebook

### 5. Launch the app

```
python skin_cancer_app.py
```

Open browser at: http://127.0.0.1:7860

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Gradio
- Cohere API
- python-dotenv
- NumPy / Pandas
- Jupyter Notebook

---

## 📊 Model Performance

- Dataset: ISIC Skin Cancer (3000 balanced images)
- Training Accuracy: ~80%
- Epochs: 3 (with early stopping)

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist.

---

## 👩‍💻 Developer

Rabia Irshad
GitHub: https://github.com/rabia-irshad2
LinkedIn: https://www.linkedin.com/in/rabiairshad2
