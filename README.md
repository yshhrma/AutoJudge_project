# ⚖️ AutoJudge – Automated Programming Problem Evaluation System

## 📌 Project Overview
AutoJudge is an end-to-end **machine learning–based evaluation system** designed to automatically analyze programming problems and predict:

- **Difficulty Level**: Easy / Medium / Hard (Classification)
- **Problem Score**: A continuous numerical score (Regression)

The system processes the textual description of a programming problem and uses Natural Language Processing (NLP) and machine learning models to infer its complexity and difficulty. A **Streamlit-based web interface** is provided for interactive usage.

---

## 📂 Dataset Used
The dataset consists of programming problems represented in **JSONL format**, where each entry includes:

- Problem title
- Problem description
- Input description
- Output description
- Difficulty class label
- Numerical problem score

**Note:**  

[The dataset used for this project is as same as what is provided.]

---

## 🧠 Approach and Models Used

### 🔹 Text Preprocessing
- Lowercasing text
- Removing special characters
- Stopword removal using **NLTK**
- Combining all text fields into a single context

### 🔹 Feature Engineering
- **TF-IDF Vectorization** (top features selected)
- Manual features:
  - Count of mathematical symbols (`$`)
  - Text length of the combined description

### 🔹 Machine Learning Models

#### Classification (Difficulty Prediction)
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

The best-performing classifier is automatically selected based on validation accuracy.

#### Regression (Score Prediction)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best regressor is chosen based on lowest RMSE.

---

## 📊 Evaluation Metrics

**Classification Metrics**
- Accuracy
- Precision
- Recall
- F1-score

**Regression Metrics**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Training Logistic Regression...
  -> Accuracy: 0.5237
Training Random Forest...
  -> Accuracy: 0.5310
Training SVM...
  -> Accuracy: 0.5103
 
 Best Classification Model: Random Forest (Accuracy: 0.5310)

--- Detailed Classification Report ---
              precision    recall  f1-score   support

        Easy       0.55      0.21      0.30       136
        Hard       0.58      0.85      0.69       425
      Medium       0.32      0.19      0.24       262

    accuracy                           0.53       823
   macro avg       0.48      0.41      0.41       823
weighted avg       0.49      0.53      0.48       823
---

Confusion Matrix:
[[ 28  61  47]
 [  9 360  56]
 [ 14 199  49]]

Training Linear Regression...
  -> RMSE: 7.0971 | MAE: 5.4617
Training Random Forest Regressor...
  -> RMSE: 2.0396 | MAE: 1.6897
Training Gradient Boosting...
  -> RMSE: 2.0338 | MAE: 1.6976

🏆 Best Regression Model: Gradient Boosting (RMSE: 2.0338)

## 🚀 Steps to Run the Project Locally

### 1️⃣ Clone the Repository
! git clone https://github.com/yshhrma/AutoJudge_project.git
% cd AutoJudge_project

### 2️⃣ Install Dependencies
! pip install -r requirements.txt

### 3️⃣ Preprocess the Dataset
python preprocess.py

### 4️⃣ Feature Extraction
python feature_extraction.py

### 5️⃣ Train Models
python train_models.py

### 6️⃣ Run the Web Interface
streamlit run app.py

### 🌐 Web Interface Explanation
The web interface is built using Streamlit and allows users to:
Enter a programming problem description
Provide input and output format details
Click a button to predict:
Difficulty level (Easy / Medium / Hard)
Numerical problem score
The interface internally applies the same preprocessing and feature extraction steps used during training, ensuring consistent and reliable predictions.

### 🎥 Demo Video
Link:- https://youtu.be/qt6HLdwJNYo?si=_yjPoleBbARFKBxJ

### 👤 Author Details
Name: Yash Sharma
Enrollment Number: 24116108
Branch: Electronics and Communication Engineering (ECE)
Email: yash_s@ece.iitr.ac.in
Contact Number: 8209788608
