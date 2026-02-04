import pandas as pd
from sklearn.metrics import classification_report,roc_auc_score
import joblib

model = joblib.load("model.pkl")

df = pd.read_csv("../data/heart.csv")

X = df.drop("target",axis=1)
y = df["target"]

pred = model.predict(X)
probs = model.predict_proba(X)[:,1]

print(classification_report(y,pred))
print("ROC AUC:",roc_auc_score(y,probs))
