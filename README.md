# Pet-Face-Classification
# 🐾 Pets Facial Expression Classifier

> A deep learning model that reads the emotions on your pet's face — built with TensorFlow/Keras and a custom CNN architecture.
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-Specify-lightgrey)

---

## 🌟 Overview
Ever wondered if your dog looks *happy*, *sad*, or *angry*? This project trains a Convolutional Neural Network (CNN) from scratch to classify pet facial expressions using the [Pets Facial Expression dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) from Kaggle.

The model takes in images of pet faces and predicts their emotional expression class — no pretrained backbone required, just a deep custom architecture trained end-to-end.

---

## ✨ Features
- 📥 **Auto dataset download** via `kagglehub` — no manual setup needed
- 🧠 **Custom deep CNN** with 4 convolutional blocks (64 → 512 filters)
- 🖼️ **Automatic train/validation split** (80/20) straight from the image directory
- 📊 **Built-in visualizations** — sample image grid + accuracy/loss curves
- 💾 **Model export** to `.h5` for easy reuse or deployment
- 🎯 **Multi-class classification** with softmax output, adaptable to any number of expression classes

---

## 🗂️ Project Structure
```
├── pet_facial_expression_classifier.py   # Main training script
├── unique_pet_face_classifier.h5         # Saved trained model (generated after running)
└── README.md
```

---

## 🧰 Requirements
- Python 3.8+
- TensorFlow (2.x)
- Keras (bundled with TensorFlow)
- matplotlib
- numpy
- kagglehub

Install everything with:

```bash
pip install tensorflow matplotlib numpy kagglehub
```

---

## 🔑 Kaggle API Setup
This project uses `kagglehub` to automatically pull the dataset, which requires Kaggle API credentials.

1. Create a Kaggle account and generate an API token from your [Kaggle account settings](https://www.kaggle.com/settings).
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

---

## 🚀 Usage
Run the training script:

```bash
python pet_facial_expression_classifier.py
```

### What happens when you run it:
1. 📦 Downloads the Pets Facial Expression dataset from Kaggle
2. 🏷️ Automatically infers class labels from the folder structure
3. ✂️ Splits data into 80% training / 20% validation
4. 🖼️ Displays a 3×3 grid of sample images with their labels
5. 🔢 Normalizes pixel values to the `[0, 1]` range
6. 🏗️ Builds and compiles a custom 4-block CNN
7. 🏋️ Trains the model for 5 epochs
8. 📈 Plots training/validation accuracy and loss curves
9. 💾 Saves the trained model as `unique_pet_face_classifier.h5`

---

## 🏗️ Model Architecture
| Layer | Details |
|---|---|
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 256 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 512 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 1024 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | `len(class_names)` units, Softmax |

**Compilation settings:**
- **Optimizer:** Adam (learning rate = `0.0001`)
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
---

## ⚙️ Configuration
| Parameter | Location | Default | Description |
|---|---|---|---|
| `image_size` | `image_dataset_from_directory(...)` | `(150, 150)` | Input image dimensions |
| `batch_size` | `image_dataset_from_directory(...)` | `32` | Batch size for training/validation |
| `validation_split` | `image_dataset_from_directory(...)` | `0.2` | Fraction of data held out for validation |
| `epochs` | `model.fit(...)` | `5` | Number of training epochs |
| `learning_rate` | `Adam(learning_rate=...)` | `0.0001` | Optimizer learning rate |

---

## 📌 Notes & Limitations
- Only **5 epochs** are used by default — increase this for better convergence and higher accuracy.
- The model is trained **from scratch** (no transfer learning), so it may need more data/epochs to match the performance of a pretrained backbone like ResNet or MobileNet.
- Class names and their count are inferred automatically from the dataset's folder structure — no manual label mapping is needed.
- `model.save(...)` uses the legacy `.h5` format; consider switching to the native Keras `.keras` format for newer TensorFlow versions.

---

## 🔮 Future Improvements
- [ ] Add data augmentation (rotation, flip, zoom) to reduce overfitting
- [ ] Experiment with transfer learning (e.g., MobileNetV2, EfficientNet) for higher accuracy
- [ ] Add early stopping and model checkpointing
- [ ] Add a confusion matrix and per-class precision/recall report
- [ ] Build a simple inference script/demo for predicting on new pet images
- [ ] Export to `.keras` or TensorFlow Lite for lightweight deployment

---

## 📄 License
Specify your license here (e.g., MIT, Apache 2.0).

---

## 🙏 Acknowledgments

- **Dataset:** [Pets Facial Expression Dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) by Ansh Tanwar on Kaggle
- **Framework:** [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/)

---

<p align="center">Made with 🐶🐱 and a lot of convolutions</p>
