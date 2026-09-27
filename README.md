# Heart-Disease-Detectection-Using-Machine-Learning-Pipeline
deployment link:https://heart-disease-detectection-using-machine-4rp6.onrender.com

# ❤️ Heart Disease Prediction — Machine Learning & Flask Web App

A machine learning project that predicts the presence of heart disease from patient clinical features. The project includes a complete ML preprocessing pipeline, multiple classification algorithms, model evaluation, hyperparameter tuning, and a Flask-based web application for real-time prediction.

> **⚕️ Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Heart disease prediction is a binary classification problem where the model predicts whether a patient profile indicates:

* `0` → Normal / No Heart Disease
* `1` → Heart Disease

The project uses the **UCI Heart Disease dataset** and applies several machine learning preprocessing and modeling techniques before deploying a trained **Gaussian Naive Bayes** model through Flask.

The web application allows users to enter seven selected clinical features and receive a prediction of **"Normal"** or **"Heart Disease Detected"**.

---

## 🚀 Key Features

* 📊 Exploratory data preprocessing
* 🔍 Missing-value checking
* 🔄 Yeo-Johnson variable transformation
* ✂️ Feature selection
* ⚖️ Class balancing using SMOTE
* 📏 Feature scaling using StandardScaler
* 🤖 Multiple classification algorithms
* 🎯 Hyperparameter tuning using GridSearchCV
* 📈 Accuracy, Confusion Matrix and Classification Report
* 📉 ROC/AUC analysis
* 💾 Model serialization using Pickle
* 🌐 Flask web application
* 🎨 Responsive HTML/CSS interface
* 🔮 Real-time prediction from user input

---

## 🧠 Machine Learning Pipeline

The project follows this workflow:

```text
Raw Dataset
     ↓
Train-Test Split
     ↓
Missing Value Check
     ↓
Numerical / Categorical Analysis
     ↓
Yeo-Johnson Transformation
     ↓
Outlier Trimming
     ↓
Feature Selection
     ↓
SMOTE Class Balancing
     ↓
Standard Scaling
     ↓
Train Multiple ML Models
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Select Trained Model
     ↓
Save Model
     ↓
Flask Deployment
```

The training code performs an 80/20 train-test split and subsequently applies the preprocessing pipeline to the training and testing data.

---

## 🔧 Data Preprocessing

### 1. Train-Test Split

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 2. Variable Transformation

The project applies **Yeo-Johnson transformation** to the input variables.

This is used to transform the distributions of numerical variables and make them more suitable for machine learning algorithms.

The implementation applies `yeojohnson` to the training and testing features and then performs outlier trimming using the IQR method.

---

### 3. Feature Selection

Feature selection is performed using techniques including:

* Constant feature detection
* Quasi-constant feature detection
* Hypothesis-testing-based feature analysis

Several low-information features are removed during this stage.

---

### 4. Class Balancing with SMOTE

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training dataset.

```python
SMOTE(random_state=42)
```

SMOTE is applied only to the training data to generate synthetic samples for the minority class.

---

### 5. Feature Scaling

The project uses:

```python
StandardScaler()
```

The scaler is fitted on the balanced training data and then used to transform both training and testing data. The fitted scaler is saved as `standerd_model.pkl`.

---

# 🤖 Machine Learning Models

The project evaluates multiple classification algorithms:

| Model               | Algorithm            |
| ------------------- | -------------------- |
| KNN                 | K-Nearest Neighbors  |
| Naive Bayes         | Gaussian Naive Bayes |
| Logistic Regression | Logistic Regression  |
| Decision Tree       | Decision Tree        |
| Random Forest       | Random Forest        |
| AdaBoost            | AdaBoost             |
| Gradient Boosting   | Gradient Boosting    |
| XGBoost             | XGBoost              |

These models are implemented and evaluated using accuracy, confusion matrix, and classification report.

---

## 🎯 KNN Hyperparameter Selection

For KNN, different values of `k` can be evaluated:

```python
for k in [1, 3, 5, 7, 11, 13]:
    model = KNeighborsClassifier(n_neighbors=k)
```

The current implementation uses:

```python
KNeighborsClassifier(n_neighbors=5)
```

The model is then evaluated on the test dataset.

---

## 🔬 Naive Bayes

The deployed model is **Gaussian Naive Bayes**.

```python
GaussianNB(var_smoothing=1e-10)
```

The trained model is saved as:

```text
navi_bayes_model.pkl
```

The training code also contains a GridSearchCV configuration for tuning `var_smoothing` using 5-fold cross-validation and accuracy scoring.

---

## 📊 Model Evaluation

The project evaluates models using:

### Accuracy

Measures the overall percentage of correct predictions.

### Confusion Matrix

Shows:

```text
True Positive
True Negative
False Positive
False Negative
```

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

The project also generates ROC curves for the evaluated models using False Positive Rate and True Positive Rate.

---

# 🌐 Flask Web Application

The trained Gaussian Naive Bayes model and StandardScaler are loaded into a Flask application.

The application accepts seven features:

```text
age
sex
cp
thalach
oldpeak
slope
thal
```

These features are maintained in a specific order because the order must match the model's training pipeline.

The Flask application:

```text
User Input
    ↓
Convert Input to Numerical Values
    ↓
StandardScaler
    ↓
Gaussian Naive Bayes
    ↓
Prediction
    ↓
Normal / Heart Disease Detected
```

The prediction logic returns either **"Normal"** or **"Heart Disease Detected"**.

---

# 🖥️ Web Interface

The application provides a user-friendly interface called **CardioCheck**.

Users can enter:

* Age
* Sex
* Chest Pain Type
* Maximum Heart Rate
* ST Depression (`oldpeak`)
* Slope
* Thalassemia result

## The interface provides the prediction together with an optional model confidence value.

# 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart.csv
├── main.py
├── app.py
│
├── all_models.py
├── feature_select.py
├── variable_transformation_tech.py
├── log_code.py
│
├── navi_bayes_model.pkl
├── standerd_model.pkl
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── Procfile
│
└── README.md
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
```

```bash
cd heart-disease-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project dependencies include Flask, NumPy, Pandas, Scikit-learn, imbalanced-learn, Matplotlib, XGBoost and related packages.

---

# ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

The Flask application is configured to run on port `5000`.

---

# 🔮 Example Prediction

Example input:

```text
Age       : 45
Sex       : Male
CP        : Non-Anginal Pain
Max HR    : 150
Oldpeak   : 1.0
Slope     : Flat
Thal      : Normal
```

The application processes the input through the saved scaler and trained Gaussian Naive Bayes model before displaying the prediction.

---

# 💾 Saved Model Files

The project saves two important artifacts:

### StandardScaler

```text
standerd_model.pkl
```

Used to transform incoming user data using the same scaling process used during model training.

### Gaussian Naive Bayes

```text
navi_bayes_model.pkl
```

Contains the trained classification model used by the Flask application.

---

# 🛠️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* Gaussian Naive Bayes
* KNN
* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

### Data Processing

* Pandas
* NumPy
* SciPy
* SMOTE
* StandardScaler
* Yeo-Johnson Transformation

### Visualization

* Matplotlib

### Deployment

* Flask
* Gunicorn

### Frontend

* HTML
* CSS
* Jinja2

---

# 📚 What I Learned

Through this project, I worked with:

* End-to-end machine learning pipelines
* Data preprocessing
* Train-test splitting
* Feature transformation
* Outlier handling
* Feature selection
* SMOTE
* Feature scaling
* Classification algorithms
* Hyperparameter tuning
* Cross-validation
* Confusion matrix
* Precision, Recall and F1-score
* ROC-AUC
* Model serialization
* Flask model deployment

---

# 🚀 Future Improvements

Possible future improvements include:

* Add more advanced hyperparameter optimization
* Compare models using additional evaluation metrics
* Improve recall for the positive class
* Add interactive prediction history
* Add database integration
* Improve model monitoring
* Add automated testing
* Containerize the application using Docker
* Deploy the application to a cloud platform

---

# ⚠️ Disclaimer

This application is an **educational machine learning project**.

It should **not** be considered a medical diagnostic tool. Predictions generated by this application should not be used to make medical decisions. Always consult a qualified healthcare professional for medical concerns.

---

# 👨‍💻 Author

**Sagar Datti**

B.Tech — Computer Science & Engineering (Data Science)

---

⭐ If you found this project useful, consider giving the repository a star!
