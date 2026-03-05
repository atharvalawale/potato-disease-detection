🥔 Potato Leaf Disease Detection using CNN
A deep learning project to detect diseases in potato plant leaves using a Convolutional Neural Network (CNN). Built as part of a team of 4 for academic research, with findings presented at ICNGT-2025 (International Conference on Next Generation Technologies for Sustainable Development), organized by SVKM's NMIMS University.

📌 Problem Statement
Potato crops are highly vulnerable to diseases like Early Blight and Late Blight, which cause significant agricultural and economic losses every year. Traditional disease identification is slow, manual, and error-prone. This project aims to automate disease detection using deep learning to enable early intervention and protect crop yield.

🎯 Objective
To build a CNN-based image classification model that can accurately classify potato leaf images into:

🟡 Early Blight
🔴 Late Blight
🟢 Healthy


📊 Model Performance
MetricScoreTraining Accuracy95%Validation Accuracy90%Classes3 (Early Blight, Late Blight, Healthy)

🛠️ Tech Stack

Language: Python
Framework: TensorFlow / Keras
Model: Convolutional Neural Network (CNN)
Libraries: NumPy, Matplotlib, OpenCV, scikit-learn
Dataset: PlantVillage Dataset


🗂️ Dataset
The model was trained on the publicly available PlantVillage Dataset, which contains labeled images of potato leaves across three categories: Early Blight, Late Blight, and Healthy.

🧠 Model Architecture
The CNN model consists of:

Input layer with image resizing and rescaling
Multiple Convolutional layers with ReLU activation
MaxPooling layers for spatial downsampling
Data Augmentation (random flip, random rotation)
Dense layers with Dropout for regularization
Softmax output layer for 3-class classification

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Evaluation Metric: Accuracy, Precision, Recall, F1-Score

📁 Project Structure
potato-disease-detection/
│
├── data/                   # Dataset directory
├── notebooks/              # Jupyter notebooks for EDA & training
├── model/                  # Saved model files
├── src/
│   ├── preprocess.py       # Data loading & augmentation
│   ├── model.py            # CNN architecture
│   └── evaluate.py         # Evaluation metrics
├── requirements.txt
└── README.md

🚀 How to Run

Clone the repository

bashgit clone https://github.com/atharvalawale/potato-disease-detection.git
cd potato-disease-detection

Install dependencies

bashpip install -r requirements.txt

Run the notebook

bashjupyter notebook notebooks/potato_disease_detection.ipynb

📈 Results
The model successfully classifies potato leaf diseases with:

95% training accuracy
90% validation accuracy

Evaluation using precision, recall, and F1-score confirmed reliable classification performance on the imbalanced dataset.

👥 Team
Developed by a team of 4 students from the Computer Engineering Department, K.J. Somaiya Institute of Technology, Mumbai.

📄 Research
This project was accepted and presented at:

ICNGT-2025 — International Conference on Next Generation Technologies for Sustainable Development
Organized by SVKM's NMIMS University, Shirpur Campus | March 28–29, 2025


📬 Contact
Atharva Lawale
📧 atharvalawale383@gmail.com
🔗 LinkedIn
🐙 GitHub
