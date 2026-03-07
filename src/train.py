import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import joblib

from features import build_features

matches = pd.read_csv("data/Matches.csv", low_memory=False)

df = build_features(matches)

train = df[df["MatchDate"] < "2020-01-01"]
test = df[df["MatchDate"] >= "2020-01-01"]

X_train = train[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
X_test = test[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
y_train = train["Result"]
y_test = test["Result"]

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


y_pred = log_reg.predict_proba(X_test)[:,1]
y_class = (y_pred >= 0.5).astype(int)


print("Log Reg model:")
print("Log loss:", log_loss(y_test,y_pred))
print("AUC: ", roc_auc_score(y_test,y_pred))
print("Accuracy: ", accuracy_score(y_test,y_class))

joblib.dump(log_reg, "models/model_lr.pkl")


rf = RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=50,random_state=42,n_jobs=1)

rf.fit(X_train,y_train)

rf_pred = rf.predict_proba(X_test)[:,1]
rf_class = (rf_pred >= 0.5).astype(int)

print("\nRandom Forest model:")
print("Log loss: ", log_loss(y_test, rf_pred))
print("AUC: ", roc_auc_score(y_test, rf_pred))
print("Accuracy: ", accuracy_score(y_test, rf_class))

joblib.dump(rf, "models/model_rf.pkl")


xgb = XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=4)
xgb.fit(X_train,y_train)

xgb_pred = xgb.predict_proba(X_test)[:,1]
xgb_class = (xgb_pred >= 0.5).astype(int)

print("\nXGBoost model:")
print("Log loss: ", log_loss(y_test, xgb_pred))
print("AUC: ", roc_auc_score(y_test, xgb_pred))
print("Accuracy: ", accuracy_score(y_test, xgb_class))

joblib.dump(xgb, "models/model_xgb.pkl")
