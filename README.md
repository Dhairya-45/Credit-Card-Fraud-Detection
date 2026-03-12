# 🛡️ Credit Card Fraud Detection

A machine learning web application that detects fraudulent credit card transactions in real time using a trained ML model deployed with Streamlit.

---

## 🔗 Live Demo
👉 [Click here to open the app](https://your-app-link.streamlit.app)

---

## 📌 Project Overview

Credit card fraud is a major problem worldwide. This project uses machine learning to classify transactions as **fraudulent** or **legitimate** based on transaction features.

The app allows:
- Single transaction prediction with instant result
- Batch prediction by uploading a CSV file
- Adjustable detection threshold using a slider

---

## 🖼️ App Preview

| Single Prediction | Batch Prediction |
|---|---|
| Enter transaction features manually | Upload CSV and get results for all rows |
| Shows fraud probability | Shows total fraud count and flagged transactions |

---

## 🗂️ Project Structure

```
Credit-Card-Fraud-Detection/
├── pkl/
│   ├── fraud_model.pkl          # Trained ML model
│   └── fraud_scaler.pkl         # Fitted scaler for Amount feature
├── Credit_Card_Fraud_Detection.ipynb   # Model training notebook
├── streamlit_app.py             # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Model Details

- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records:** 284,807 transactions
- **Fraud Cases:** 492 (0.17% — highly imbalanced)
- **Features:** Time, V1–V28 (PCA components), Amount
- **Target:** Class (0 = Legitimate, 1 = Fraud)

---

## ⚙️ How It Works

1. User enters transaction features (Time, V1–V28, Amount)
2. The `Amount` feature is scaled using the saved scaler
3. The trained model predicts fraud probability
4. If probability ≥ threshold → **FRAUD**, else → **LEGITIMATE**

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Dhairya-45/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run streamlit_app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
numpy
pandas
scikit-learn
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~99% |
| Precision | High |
| Recall | High |
| Model Type | Random Forest / Logistic Regression |

---

## 👨‍💻 Author

**Dhairya**
- GitHub: [@Dhairya-45](https://github.com/Dhairya-45)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).