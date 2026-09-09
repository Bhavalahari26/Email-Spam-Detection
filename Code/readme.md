# 📧 SMART Email Spam Detection System

A machine learning-based email spam detection system that uses **semantic analysis, adversarial training, and reinforcement learning** to classify emails as **Spam** or **Ham (Not Spam)**.

The project provides a web-based interface where users can register, log in, enter email content, and receive a spam prediction.

---

## 🚀 Project Overview

Email spam is one of the most common problems in digital communication. Traditional spam filters may struggle with sophisticated and continuously changing spam messages.

This project, **SMART – Semantic, Multi-Objective, and Reinforcement-Based Adversarial Training for Email Spam Detection**, combines multiple machine learning and NLP techniques to improve spam detection.

The system analyzes email text and predicts whether the message is:

- 🟢 **Ham** – Legitimate email
- 🔴 **Spam** – Unwanted or malicious email

---

## ✨ Features

- 📧 Email spam/ham classification
- 🧠 NLP-based text processing
- 🔍 Semantic feature extraction
- 🤖 Machine learning-based prediction
- 🛡️ Adversarial training
- 🎯 Reinforcement learning-based optimization
- 📊 Analytics dashboard
- 🔐 User registration and login
- 🌐 Web-based prediction interface
- 📱 Responsive frontend interface

---

## 🧠 Machine Learning Approach

The project combines multiple approaches for email spam detection.

### 1. Text Preprocessing

Email text is cleaned and transformed into numerical representations suitable for machine learning.

### 2. TF-IDF Vectorization

TF-IDF is used to represent the importance of words in email messages.

### 3. Semantic Representation

Word embeddings and transformer-based techniques are used to capture the semantic meaning of email content.

### 4. Adversarial Training

Adversarial examples are used to improve the robustness of the spam detection model against modified or deceptive spam messages.

### 5. Reinforcement Learning

A reinforcement learning agent is used to optimize the detection process and improve model performance.

---

## 🛠️ Technologies Used

### Programming Languages

- Python
- HTML
- CSS
- JavaScript

### Machine Learning & NLP

- TensorFlow
- Keras
- PyTorch
- Scikit-learn
- BERT
- Word2Vec
- TF-IDF

### Backend

- Flask
- Python

### Frontend

- HTML5
- CSS3
- JavaScript

### Database

- SQLite

### Development Tools

- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```text
Email-Spam-Detection/
│
└── Code/
    │
    ├── app.py
    ├── Train.py
    ├── emails.csv
    ├── readme.md
    │
    ├── models/
    │   ├── smart_adversarial_model.h5
    │   ├── smart_baseline_model.pkl
    │   ├── smart_rl_agent.pth
    │   └── tfidf_vectorizer.pkl
    │
    ├── static/
    │   └── css/
    │       └── style.css
    │
    └── templates/
        ├── analytics.html
        ├── base.html
        ├── basepaper.html
        ├── landing.html
        ├── login.html
        ├── predict.html
        └── register.html
