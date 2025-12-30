import joblib
import pandas as pd
from features import build_features

def predict_matches(path):
    model = joblib.load("model_lr.pkl")
    matches = pd.read_csv(path, low_memory=False)

    features = build_features(matches)
    
    X = features[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
    probs = model.predict_proba(X)[:,1]

    features = features.assign(HomeWinProb=probs)
    return features[["MatchDate","HomeWinProb"]]

if __name__ == "__main__":
    path = input("Input the name file in the /data folder: ")
    path = "data/" + path + ".csv"
    preds = predict_matches(path)
    print(preds.head())
