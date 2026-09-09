from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib, torch
from tensorflow.keras.models import load_model
import torch.nn as nn

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default="user")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
baseline_model = joblib.load("models/smart_baseline_model.pkl")
nn_model = load_model("models/smart_adversarial_model.h5")

class SpamAgent(nn.Module):
    def __init__(self, input_dim=5000):
        super(SpamAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=1)

agent = SpamAgent()
agent.load_state_dict(torch.load("models/smart_rl_agent.pth"))
agent.eval()

def predict_email(text, model_type="nn"):
    X_vec = vectorizer.transform([text])
    if model_type == "baseline":
        return baseline_model.predict(X_vec)[0]
    elif model_type == "nn":
        pred = nn_model.predict(X_vec.toarray())
        return int(pred[0][0] > 0.5)
    elif model_type == "rl":
        X_torch = torch.tensor(X_vec.toarray(), dtype=torch.float32)
        with torch.no_grad():
            outputs = agent(X_torch)
            _, pred = torch.max(outputs, 1)
        return int(pred.item())

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        user = User.query.filter_by(username=u).first()
        if user and check_password_hash(user.password, p):
            session["user"] = u
            session["role"] = user.role
            return redirect(url_for("predict"))
        else:
            flash("Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        role = request.form.get("role", "user")
        hashed = generate_password_hash(p)
        db.session.add(User(username=u, password=hashed, role=role))
        db.session.commit()
        flash("Registered successfully")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        email = request.form["email"]
        model_type = request.form["model"]
        result = predict_email(email, model_type)
    return render_template("predict.html", result=result)

@app.route("/analytics")
def analytics():
    if "role" in session and session["role"] == "admin":
        return render_template("analytics.html")
    return redirect(url_for("landing"))

@app.route("/basepaper")
def basepaper():
    return render_template("basepaper.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
