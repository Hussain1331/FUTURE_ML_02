# 🎫 Support Ticket Classification System

An AI-powered customer support ticket classification system built using **Machine Learning, NLP, Streamlit, and Scikit-Learn**.

The application automatically analyzes customer support tickets, predicts the appropriate support queue, and assigns a priority level to help support teams route tickets efficiently.

---

## 🚀 Features

* Automatic Support Ticket Classification
* Priority Level Prediction (High, Medium, Low)
* Natural Language Processing (NLP) Pipeline
* TF-IDF Feature Extraction
* Real-Time Predictions with Streamlit
* Modern Professional Dashboard UI
* Queue Routing Automation
* Text Preprocessing using NLTK

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* LinearSVC
* Logistic Regression
* Multinomial Naive Bayes

### NLP

* NLTK
* Tokenization
* Stopword Removal
* Lemmatization
* Text Cleaning

### Deployment

* Streamlit

### Data Processing

* Pandas
* NumPy

---

## 📂 Dataset Features

The dataset contains customer support tickets with the following attributes:

| Feature  | Description                 |
| -------- | --------------------------- |
| Subject  | Ticket title                |
| Body     | Ticket description          |
| Queue    | Support department/category |
| Priority | Ticket urgency level        |
| Language | Ticket language             |

---

## 🔄 Machine Learning Workflow

### 1. Data Loading

* Imported customer support ticket dataset using Pandas.

### 2. Data Cleaning

* Removed unnecessary columns.
* Handled missing values.
* Removed duplicate records.
* Filtered English-language tickets.

### 3. Text Preprocessing

* Converted text to lowercase.
* Removed special characters and punctuation.
* Tokenized text.
* Removed stopwords.
* Applied lemmatization.

### 4. Feature Engineering

* Used TF-IDF Vectorization.
* Generated numerical text representations.

### 5. Label Encoding

* Encoded queue categories.
* Encoded priority levels.

### 6. Model Training

Three machine learning algorithms were evaluated:

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Classifier (LinearSVC)

### 7. Model Evaluation

The best-performing model was selected based on accuracy scores.

---

## 📊 Model Performance

### Queue Classification

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 41.00%   |
| Logistic Regression | 49.05%   |
| LinearSVC           | 57.47%   |

### Priority Prediction

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 50.21%   |
| Logistic Regression | 57.83%   |
| LinearSVC           | 59.64%   |

### Best Model

**LinearSVC**

---

## 🧠 Prediction Outputs

### Input

Customer Support Ticket

Example:

"I have been charged twice for my subscription and need a refund immediately."

### Output

Queue:
Billing and Payments

Priority:
High

---

## 📁 Project Structure

Support-Ticket-Classifier/

├── models/

│ ├── queue_model.pkl

│ ├── priority_model.pkl

│ ├── tfidf.pkl

│ ├── queue_encoder.pkl

│ └── priority_encoder.pkl

│

├── app.py

├── requirements.txt

├── README.md

└── notebook.ipynb

---



## 🎯 Future Improvements

* Deep Learning Models (LSTM, BERT)
* Confidence Score Display
* Multi-language Support
* Ticket Analytics Dashboard
* Automated Email Routing
* Explainable AI Predictions

---
This project was developed as part of a Machine Learning Internship to automate customer support ticket triaging using Natural Language Processing and Machine Learning techniques.
