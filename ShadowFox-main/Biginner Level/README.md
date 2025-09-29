# 🖼️ CIFAR-10 Image Classification using CNN  

## 📌 Project Overview  
This project demonstrates **image classification** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  
The model is trained on the **CIFAR-10 dataset**, which contains **60,000 color images** (32×32 pixels) in **10 different classes**.  

It is a **beginner-friendly deep learning project** that helps understand how CNNs work for computer vision tasks.  

---

## 🎯 Objectives  
- Learn how to preprocess and normalize image datasets.  
- Build a custom CNN model from scratch.  
- Train the model using supervised learning.  
- Evaluate accuracy on unseen test data.  
- Provide a foundation for more advanced image classification tasks.  

---

## 🗂️ Dataset – CIFAR-10  
- **Training set:** 50,000 images  
- **Test set:** 10,000 images  
- **Image size:** 32×32 pixels (RGB)  
- **Classes (10):**  
  1. Airplane ✈️  
  2. Automobile 🚗  
  3. Bird 🐦  
  4. Cat 🐱  
  5. Deer 🦌  
  6. Dog 🐶  
  7. Frog 🐸  
  8. Horse 🐎  
  9. Ship 🚢  
  10. Truck 🚚  

---

## 🧠 Model Architecture  
The CNN model is built using the **Keras Functional API**:  

1. **Input Layer:** Shape (32, 32, 3)  
2. **Conv Block 1:**  
   - Conv2D (32 filters, 3×3, ReLU) → BatchNorm  
   - Conv2D (32 filters, 3×3, ReLU) → BatchNorm  
   - MaxPooling2D (2×2)  
   - Dropout (0.2)  

3. **Conv Block 2:**  
   - Conv2D (64 filters, 3×3, ReLU) → BatchNorm  
   - Conv2D (64 filters, 3×3, ReLU) → BatchNorm  
   - MaxPooling2D (2×2)  
   - Dropout (0.2)  

4. **Fully Connected Layers:**  
   - Flatten  
   - Dense (1024 neurons, ReLU) → Dropout (0.2)  
   - Dense (10 neurons, Softmax) → Output  

---

## ⚙️ Techniques Used  
- **Supervised Learning**  
- **Convolutional Neural Networks (CNNs)**  
- **Batch Normalization** (stabilizes learning)  
- **Dropout** (reduces overfitting)  
- **MaxPooling** (downsamples feature maps)  
- **Softmax Activation** (multi-class classification)  
- **Adam Optimizer** (adaptive learning rate)  
- **Sparse Categorical Crossentropy Loss**  

---

## 📊 Training  
- **Epochs:** 50  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metric:** Accuracy  

---

## 📈 Results  
- The model achieves around **70–80% test accuracy** (depending on training).  
- Accuracy can be improved with:  
  - More training epochs  
  - Data augmentation  
  - Pre-trained models (VGG, ResNet, etc.)  

---

## 🌐 Live Web App  
👉 [Click here to try the app](https://prediction-bauz83hmln4cshc9tfjb7q.streamlit.app/)

---

## 🚀 How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/cifar10-cnn.git
   cd cifar10-cnn
   ```
2. Install dependencies:  
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. Run the notebook:  
   ```bash
   jupyter notebook Biginner_level.ipynb
   ```

---

## 📌 Future Improvements  
- Use **Data Augmentation** to make the model more robust.  
- Try **transfer learning** with pretrained models like **ResNet, VGG16, MobileNet**.  
- Deploy the model using **Flask/Streamlit** for a web demo.  

---

## 🏆 Skills Gained  
- TensorFlow/Keras  
- CNNs for Computer Vision  
- Image Preprocessing  
- Model Training & Evaluation  
- Overfitting Prevention (Dropout, BatchNorm)  

---

## 📬 Author  
👤 **Syed Nabeel Ahmed**  
- [GitHub](https://github.com/nabeelsyed11)  
- [LinkedIn](https://www.linkedin.com/in/syed-ahmed-64052a32a/)  

---

✨ *This project is a beginner-friendly introduction to Deep Learning for Computer Vision using CNNs.*  
