# CANTILEVER-Fraud

## 📌 Project Overview
CANTILEVER-Fraud is a machine learning project focused on **credit card fraud detection**.  
The dataset is highly imbalanced (fraudulent transactions are less than 0.2% of all data).  
The project explores different techniques such as **Logistic Regression**, **Random Forest**,  
and strategies like **class weighting** and **SMOTE** to handle imbalance.

---

## 📂 Project Structure
CANTILEVER-Fraud/
│
├── data/ # Raw dataset (creditcard.csv from Kaggle)
│
├── notebooks/ # Jupyter notebooks
│ ├── fraud_eda.ipynb
│ ├── fraud_baseline.ipynb
│ └── fraud_final.ipynb
│
├── models/
│ └── sklearn/
│ └── fraud_pipeline.joblib
│
├── src/
│ └── fraud_predict.py
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup & Installation
```bash
# Clone repository
git clone git@github.com:benjakariti/CANTILEVER-Fraud.git
cd CANTILEVER-Fraud

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
🚀 Usage
Run inference on new data:

bash
Copy code
python src/fraud_predict.py --input data/sample.csv
📊 Models
Logistic Regression (with class weighting)

Random Forest (with class weighting / SMOTE)

Final pipeline saved in models/sklearn/fraud_pipeline.joblib

👤 Author
Benjamin Kariti