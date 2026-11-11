📉 Customer Churn Prediction: End-to-End Machine Learning Project

Predicts which telecom customers are likely to cancel their service, so a business can target retention offers before they leave. Covers data cleaning, feature engineering, model comparison, hyperparameter tuning, threshold optimisation, explainability and a Streamlit app.

Tech stack: Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

🎯 Problem Statement

Acquiring a new customer costs far more than keeping an existing one. Given a customer's account details, services and billing information, can we predict whether they will churn, and what drives that decision?

📊 Dataset
Source: IBM Telco Customer Churn (Kaggle)
Size: 7,043 customers, 21 columns
Target: Churn (Yes/No), with 26.5% of customers churning (imbalanced classes)
Features: demographics, account tenure, contract type, internet and phone services, billing and payment details
🔧 Methodology
Data cleaning: TotalCharges held blank text values for brand-new customers (tenure = 0); converted to numeric and filled using MonthlyCharges × tenure. Encoded the target and dropped the ID column.
Feature engineering:
AvgMonthlySpend: total charges divided by tenure
NumServices: count of add-on services used
TenureGroup: tenure bucketed into 0–1, 1–2, 2–4 and 4–6 years
Preprocessing: scikit-learn ColumnTransformer (standard scaling + one-hot encoding) inside a Pipeline, so preprocessing is fitted only on training data (no data leakage).
Model comparison: Logistic Regression, Random Forest and Gradient Boosting, evaluated with 5-fold stratified cross-validation on ROC-AUC.
Hyperparameter tuning: RandomizedSearchCV on Gradient Boosting (15 iterations, 5-fold CV).
Threshold analysis: compared decision thresholds to understand the precision–recall trade-off for retention targeting.
Explainability: permutation importance on the held-out test set.
Deployment: Streamlit app for real-time churn risk scoring of a single customer.
📈 Results
Model comparison (5-fold CV, ROC-AUC)
Model	CV ROC-AUC
Logistic Regression	0.8457
Random Forest	0.8277
Gradient Boosting	0.8477

Best tuned parameters: n_estimators=200, learning_rate=0.02, max_depth=3, subsample=0.7

Final performance (held-out test set, 20% of data)

Test ROC-AUC: 0.846

Decision threshold	Precision	Recall	F1
0.50 (default)	0.681	0.497	0.575
0.106 (recall-focused)	0.405	0.955	0.569

Trade-off: the default threshold is precise but misses about half of the churners. Lowering it to 0.106 catches 95.5% of churners but flags a large share of all customers, which would make a retention campaign expensive. In practice the threshold should be chosen using the cost of a retention offer versus the value of a retained customer.

Top churn drivers (permutation importance)
Contract type
Tenure
Internet service
Online security
Tech support
Evaluation plots

Evaluation

Feature importance

💡 Key Takeaways
Logistic Regression performed almost as well as Gradient Boosting (0.8457 vs 0.8477), so on this dataset the simpler, more interpretable model is a strong baseline.
Choosing the decision threshold changes the business outcome far more than the choice of model.
Contract type and customer tenure are the strongest signals of churn, so retention efforts could focus on these segments.
🚀 How to Run
bash
# 1. Clone the repo
git clone https://github.com/Pritisingh06/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and save it as data/telco.csv

# 4. Train the model (creates the outputs/ folder)
python train.py

# 5. Launch the app
streamlit run app.py
📁 Project Structure
customer-churn-prediction/
├── data/              # place telco.csv here (not included)
├── outputs/           # metrics, plots and the trained model
├── train.py           # full training pipeline
├── app.py             # Streamlit app
├── requirements.txt
└── README.md
🔮 Future Improvements
Handle class imbalance with SMOTE and compare against class weights
Try XGBoost / LightGBM and add SHAP explanations
Choose the threshold using an estimated cost-benefit model
Deploy the app on Streamlit Community Cloud
