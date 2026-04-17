# 🌿 Plant Disease Detection using Deep Learning
   
A computer vision project that detects plant diseases from leaf images using **Transfer Learning** with **MobileNetV2**.   

---

## 📌 Project Overview
    
| Item | Details | 
|------|---------|
| **Task** | Multi-class Image Classification |   
| **Dataset** | PlantVillage (54,000+ images, 38 classes) |              
| **Model** | MobileNetV2 (Transfer Learning) |      
| **Framework** | TensorFlow / Keras |       
| **Accuracy** | ~90%+ on validation set |    

---
## Output

<img width="1448" height="666" alt="image" src="https://github.com/user-attachments/assets/849708cf-50d3-4bc4-af5d-1cc36bdd7c9f" />   

<img width="751" height="362" alt="image" src="https://github.com/user-attachments/assets/c5a352d7-21f5-40d1-bd1b-61a54c409a36" />

<img width="1345" height="745" alt="image" src="https://github.com/user-attachments/assets/c143514e-a67b-4051-8a30-9edbe932b45f" />

<img width="1353" height="622" alt="image" src="https://github.com/user-attachments/assets/e7ba379e-0e6d-4ba7-8e3c-767a026ad016" />

<img width="1338" height="696" alt="image" src="https://github.com/user-attachments/assets/60586c77-23bd-4fd2-8d71-ddbf09a4d956" />

---    
         
## 📁 Folder Structure
 
```
plant-disease-detection/
│
├── dataset/               ← Download PlantVillage here
│   ├── Apple___Apple_scab/
│   ├── Tomato___healthy/
│   └── ... (38 folders)
│
├── models/
│   └── plant_disease_model.h5   ← saved after training
│
├── src/
│   ├── train.py           ← Train the model
│   ├── predict.py         ← Predict on new images
│   ├── evaluate.py        ← Evaluate model performance
│   └── utils.py           ← Helper functions & class names
│
├── app/
│   └── app.py             ← Streamlit web app
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
- Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Download and extract into the `dataset/` folder
- Each subfolder = one class (e.g., `Tomato___Early_blight/`)

### Step 3 — Train the model
```bash
cd src
python train.py
```
Training takes ~30–60 mins on CPU, ~10 mins on GPU.

### Step 4 — Evaluate the model
```bash
python evaluate.py
```

### Step 5 — Predict on a new image
```bash
python predict.py --image path/to/leaf.jpg
```

### Step 6 — Launch the web app
```bash
cd app
streamlit run app.py
```

---

## 🧠 Model Architecture

```
Input Image (224 x 224 x 3)
        ↓
MobileNetV2 (pretrained on ImageNet, frozen in Phase 1)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, activation=ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(38, activation=Softmax)
        ↓
Predicted Class + Confidence
```

### Why MobileNetV2?
- Lightweight and fast (good for deployment)
- Pretrained on 1.4M ImageNet images
- Works great with small datasets via transfer learning
- Depthwise separable convolutions = fewer parameters

---

## 📊 Training Strategy

| Phase | Layers Trained | Learning Rate | Epochs |
|-------|---------------|---------------|--------|
| Phase 1 | Top layers only | 1e-3 | 10 |
| Phase 2 | Last 30 base layers | 1e-5 | 5 |

---

## 🌱 Supported Plants & Diseases (38 Classes)

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

Diseases include: Apple Scab, Black Rot, Blight, Leaf Mold, Rust, Mosaic Virus, and more.

---

## 📈 Results

After training you should see:
- **Validation Accuracy**: ~90–95%
- **Loss**: < 0.3

Check `training_curves.png` and `confusion_matrix.png` after training.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `BATCH_SIZE` to 16 in train.py |
| Slow training | Use Google Colab (free GPU) |
| Low accuracy | Train for more epochs, try EfficientNetB0 |
| Model not found | Run train.py before predict.py |

---

## 📚 References

- [PlantVillage Paper](https://arxiv.org/abs/1604.03169)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

  ---

## 📬 Contact

- 💼 LinkedIn: [linkedin.com/in/shreya-tiwari-520b692b3](https://www.linkedin.com/in/shreya-tiwari-520b692b3/)
- 📧 Email: shreyatiwari0801@gmail.com
- 🐙 GitHub: [github.com/TiwariShreya05](https://github.com/TiwariShreya05)

