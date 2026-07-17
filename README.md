# 🐾🐶🐱 Pet Facial Expression Classification using Deep Learning 🐾🐶🐱

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-red.svg">
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

# 📖 Project Overview

**Pet Facial Expression Classification** is a Deep Learning project that classifies pet facial expressions from images using a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras**.

The project automatically downloads the dataset using **KaggleHub**, preprocesses images, trains a CNN model, evaluates its performance, visualizes training results, and saves the trained model for future inference.

This project is ideal for beginners learning Computer Vision and Image Classification using TensorFlow.

---

# 🎯 Objectives

* 🐶 Classify pet facial expressions from images
* 🧠 Learn CNN-based image classification
* 📊 Visualize model performance
* 💾 Save trained deep learning model
* 🚀 Demonstrate an end-to-end TensorFlow workflow

---

# ✨ Features

✅ Automatic dataset download using KaggleHub

✅ Image preprocessing with TensorFlow

✅ Automatic training & validation split

✅ Custom Convolutional Neural Network (CNN)

✅ Multi-class image classification

✅ Accuracy & Loss visualization

✅ Trained model export (.h5)

✅ Beginner-friendly implementation

---

# 📂 Project Structure

```text
Pet-Facial-Expression-Classification/
│
├── pet_classifier.py             # Main training script
├── README.md                     # Project documentation
├── requirements.txt              # Required libraries
│
├── dataset/
│   ├── Cat/
│   ├── Dog/
│   └── ...
│
├── models/
│   └── unique_pet_face_classifier.h5
│
└── outputs/
    ├── sample_images.png
    ├── accuracy_graph.png
    └── loss_graph.png
```

---

# 🧠 Model Architecture

```text
Input Image (150×150×3)
          │
     Rescaling Layer
          │
Conv2D (64 Filters)
          │
MaxPooling
          │
Conv2D (128 Filters)
          │
MaxPooling
          │
Conv2D (256 Filters)
          │
MaxPooling
          │
Conv2D (512 Filters)
          │
MaxPooling
          │
Flatten
          │
Dense (1024)
          │
Dropout (0.5)
          │
Softmax Output
```

---

# 📦 Dataset

Dataset:

**Pets Facial Expression Dataset**

Downloaded automatically using:

```python
path = kagglehub.dataset_download(
    "anshtanwar/pets-facial-expression-dataset"
)
```

Dataset includes multiple pet facial expression classes for image classification.

---

# 🛠 Technologies Used

| Technology | Purpose                 |
| ---------- | ----------------------- |
| Python     | Programming Language    |
| TensorFlow | Deep Learning Framework |
| Keras      | Neural Network API      |
| KaggleHub  | Dataset Download        |
| NumPy      | Numerical Operations    |
| Matplotlib | Data Visualization      |
| Pillow     | Image Processing        |

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Pet-Facial-Expression-Classification.git
```

Move into the project folder

```bash
cd Pet-Facial-Expression-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

```text
tensorflow
numpy
matplotlib
kagglehub
pillow
```

Or install manually

```bash
pip install tensorflow numpy matplotlib kagglehub pillow
```

---

# ▶️ Run the Project

```bash
python pet_classifier.py
```

The program will:

* Download the dataset
* Load and preprocess images
* Display sample images
* Train the CNN model
* Validate model performance
* Plot Accuracy & Loss graphs
* Save the trained model

---

# 📊 Training Workflow

```text
Download Dataset
        │
Load Images
        │
Train / Validation Split
        │
Image Preprocessing
        │
CNN Model
        │
Model Training
        │
Validation
        │
Accuracy & Loss Graphs
        │
Save Trained Model
```

---

# 📈 Sample Training Output

```text
Epoch 1/5
Accuracy : 74.12%
Validation Accuracy : 71.85%

Epoch 2/5
Accuracy : 82.43%
Validation Accuracy : 80.26%

Epoch 3/5
Accuracy : 88.15%
Validation Accuracy : 85.47%

Epoch 4/5
Accuracy : 91.38%
Validation Accuracy : 88.71%

Epoch 5/5
Accuracy : 94.06%
Validation Accuracy : 91.24%
```

---

# 📊 Visualizations

The project generates:

* 🖼️ Sample training images
* 📈 Training Accuracy graph
* 📉 Training Loss graph

These visualizations help monitor the model's learning progress.

---

# 💾 Saved Model

The trained model is saved automatically as:

```text
unique_pet_face_classifier.h5
```

You can later load it using:

```python
from tensorflow.keras.models import load_model

model = load_model("unique_pet_face_classifier.h5")
```

---

# 🚀 Future Improvements

* ✅ Data Augmentation
* ✅ Transfer Learning (MobileNetV2, EfficientNet, ResNet50)
* ✅ Early Stopping
* ✅ Learning Rate Scheduler
* ✅ Model Checkpointing
* ✅ Confusion Matrix
* ✅ Precision, Recall & F1-Score
* ✅ TensorBoard Support
* ✅ Streamlit / Flask Web Application
* ✅ Real-time Webcam Prediction

---

# 🌟 Applications

* 🐶 Pet Emotion Recognition
* 🐱 Veterinary Assistance
* 📱 Smart Pet Monitoring Systems
* 🤖 AI-powered Animal Behavior Analysis
* 🏥 Animal Healthcare Research
* 🧠 Computer Vision Education

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Ankit Bachchhav**

AI/ML Enthusiast • Python Developer • Computer Vision Learner

* 💼 LinkedIn: https://www.linkedin.com/in/ankitbachchhav2003
* 💻 GitHub: https://github.com/yourusername

---

## ⭐ Show Your Support

If you found this project helpful, please give it a **⭐ Star** on GitHub.

Your support motivates future AI and Deep Learning projects!

---

<p align="center">
<b>🐶 Built with ❤️ using TensorFlow, Keras & Deep Learning 🚀</b>
</p>
