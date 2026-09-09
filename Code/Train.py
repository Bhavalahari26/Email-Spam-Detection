# ===========================================
# SMART Email Spam Detection (Unified Script, Clean Labels)
# ===========================================
# Implements:
#   1. Baseline classifier (LogReg)
#   2. Neural Net with Adversarial Training
#   3. Reinforcement Learning Agent (RL)
#   + Unified predict_email() for inference
# ===========================================

import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# NLTK Setup
# ----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')

# ----------------------------
# 1. Load Dataset + Handle NaN + Clean Labels
# ----------------------------
df = pd.read_csv("emails.csv")

# Drop rows with missing text or label
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

# Clean labels (support spam/ham or 0/1)
def clean_label(x):
    x = str(x).strip().lower()
    if x in ["spam", "1"]:
        return 1
    elif x in ["ham", "0"]:
        return 0
    else:
        return None

df["label"] = df["label"].apply(clean_label)
df = df.dropna(subset=["label"])  # drop bad rows
df["label"] = df["label"].astype(int)

X = df["text"]
y = df["label"]

# ----------------------------
# 2. TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ====================================================
# STEP 1: Baseline Logistic Regression
# ====================================================
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
joblib.dump(baseline_model, "smart_baseline_model.pkl")

print("\n--- Baseline Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, baseline_model.predict(X_test)))

# ====================================================
# STEP 2: Neural Net with Adversarial Training
# ====================================================

def syntactic_perturbation(text):
    noisy = []
    for c in text:
        if c.isalpha() and random.random() < 0.1:
            noisy.append(c + random.choice(["$", "#", "1"]))
        else:
            noisy.append(c)
    return "".join(noisy)

def semantic_shift(text):
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < 0.2:
            syns = wordnet.synsets(w)
            if syns:
                lemmas = syns[0].lemma_names()
                if lemmas:
                    new_words.append(random.choice(lemmas))
                    continue
        new_words.append(w)
    return " ".join(new_words)

aug_texts, aug_labels = [], []
for text, label in zip(X, y):
    aug_texts.append(text); aug_labels.append(label)
    aug_texts.append(syntactic_perturbation(text)); aug_labels.append(label)
    aug_texts.append(semantic_shift(text)); aug_labels.append(label)

X_aug = vectorizer.transform(aug_texts)
y_aug = np.array(aug_labels)

X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)

nn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_aug.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer=Adam(learning_rate=1e-3),
                 loss='binary_crossentropy', metrics=['accuracy'])

nn_model.fit(X_train_aug.toarray(), y_train_aug,
             validation_data=(X_val_aug.toarray(), y_val_aug),
             epochs=5, batch_size=128)

nn_model.save("smart_adversarial_model.h5")

loss, acc = nn_model.evaluate(X_test.toarray(), y_test)
print("\n--- Adversarially-trained Neural Net ---")
print(f"Accuracy: {acc:.4f}")

# ====================================================
# STEP 3: Reinforcement Learning Agent
# ====================================================

class SpamAgent(nn.Module):
    def __init__(self, input_dim):
        super(SpamAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=1)

X_train_torch = torch.tensor(X_train_aug.toarray(), dtype=torch.float32)
y_train_torch = torch.tensor(y_train_aug, dtype=torch.long)
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

agent = SpamAgent(input_dim=X_train_torch.shape[1])
optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_reward = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = agent(batch_x)
        _, predicted = torch.max(outputs, 1)
        rewards = torch.where(predicted == batch_y, 1.0, -1.0)
        loss = criterion(outputs, batch_y) - rewards.mean()
        loss.backward()
        optimizer.step()
        total_reward += rewards.sum().item()
    print(f"[RL Agent] Epoch {epoch+1}/5, Total Reward: {total_reward}")

torch.save(agent.state_dict(), "smart_rl_agent.pth")

# ====================================================
# Unified Prediction Function
# ====================================================

def load_models():
    global vectorizer, baseline_model, nn_model, agent
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    baseline_model = joblib.load("smart_baseline_model.pkl")
    nn_model = load_model("smart_adversarial_model.h5")

    agent = SpamAgent(input_dim=5000)
    agent.load_state_dict(torch.load("smart_rl_agent.pth"))
    agent.eval()

def predict_email(text, model_type="baseline"):
    X_vec = vectorizer.transform([text])

    if model_type == "baseline":
        pred = baseline_model.predict(X_vec)[0]
        return int(pred)

    elif model_type == "nn":
        pred = nn_model.predict(X_vec.toarray())
        return int(pred[0][0] > 0.5)

    elif model_type == "rl":
        X_torch = torch.tensor(X_vec.toarray(), dtype=torch.float32)
        with torch.no_grad():
            outputs = agent(X_torch)
            _, pred = torch.max(outputs, 1)
        return int(pred.item())

    else:
        raise ValueError("Invalid model_type. Choose 'baseline', 'nn', or 'rl'.")

# ====================================================
# Example Test
# ====================================================
if __name__ == "__main__":
    load_models()
    sample = "Congratulations! You won a free lottery ticket."
    print("\nPrediction (Baseline):", predict_email(sample, "baseline"))
    print("Prediction (NN):", predict_email(sample, "nn"))
    print("Prediction (RL):", predict_email(sample, "rl"))
