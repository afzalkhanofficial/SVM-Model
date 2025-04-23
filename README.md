# 🌊 Oil Spill Detection using Support Vector Machine (SVM)

This project implements a machine learning pipeline to classify satellite images into **oil spill** and **non-oil spill** categories using a **Support Vector Machine (SVM)**. It combines image preprocessing, feature extraction (HOG and color histograms), and optimized SVM classification to deliver accurate results.

## 📦 Technologies Used
- Python 3.x
- NumPy, OpenCV, Matplotlib
- scikit-image for HOG features
- scikit-learn for SVM and model evaluation
- TensorFlow Keras for data augmentation
- Pillow for image processing
- Joblib for model serialization

## 📁 Dataset Structure
```
SVM-Model/
├── requirements.txt
├── app.py
└── oil_spill_svm_model.pkl
```

## 🚀 Features
- Efficient image augmentation with Keras
- Preprocessing images to `128x128` resolution
- Feature extraction pipeline:
  - **HOG**: Captures texture and edge information from grayscale images
  - **Color Histograms**: Captures color distribution (32 bins per RGB channel)
- Optimized **SVM** classifier with RBF kernel and `GridSearchCV` for hyperparameter tuning
- Model persistence using Joblib
- Classification performance report (precision, recall, F1-score)

## 🛠️ Requirements
Install dependencies:
```bash
pip install numpy opencv-python scikit-image matplotlib Pillow scikit-learn tensorflow joblib
```

## 🧠 How it Works
1. **Preprocessing**: Standardize image size and apply augmentations (rotation, zoom, etc.)
2. **Feature Engineering**:
   - Convert to grayscale and extract HOG features
   - Compute normalized color histograms
3. **Model Training**:
   - Use `GridSearchCV` to tune SVM hyperparameters (`C`, `gamma`)
   - Fit the model on the training data
4. **Evaluation**:
   - Generate classification report
   - Save trained model and scaler

## 🔍 Performance
- The SVM with RBF kernel performed well on non-linear boundaries
- Tuning with GridSearchCV improved generalization and reduced overfitting

## 👤 Author
**Afzal Khan**

---

🌟 Contributions welcome! Fork this repo to adapt the SVM pipeline for other remote sensing classification tasks.
