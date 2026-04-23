# 🚀 Customer Churn Prediction System

## 📌 Overview
This project predicts whether a customer is likely to churn (leave a service) using Machine Learning.

Built using:
- Python
- Scikit-learn
- Streamlit (Web App)
- Plotly

---

## 🧠 Model
- Algorithm: Random Forest Classifier
- Dataset: Telco Customer Churn Dataset
- Accuracy: ~80%

---

## 🌐 Features
- User-friendly web interface
- Real-time churn prediction
- Probability-based output
- Business recommendations

---

## 📂 Project Structure
```
ml_project/
│
├── data/
├── model/
│ ├── churn_model.pkl
│ ├── columns.pkl
│
├── notebook/
│ └── churn_model.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py