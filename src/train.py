import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)

joblib.dump(model,"model.pkl")
print("Model saved")
