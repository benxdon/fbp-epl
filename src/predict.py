import joblib
import pandas as pd
from features import build_features
from thefuzz import fuzz, process

def process_input(inputs):

    # load matches
    matches = pd.read_csv("data/Matches.csv" , low_memory=False)

    # build hist
    home = (
        matches[["MatchDate","HomeTeam","HomeTarget","FTResult"]]
        .assign(Points=lambda x: x["FTResult"].map({"H":3,"D":1,"A":0}))
        .rename(columns={"HomeTeam":"Team","HomeTarget":"Target"})
    )

    away = (
        matches[["MatchDate","AwayTeam","AwayTarget","FTResult"]]
        .assign(Points=lambda x: x["FTResult"].map({"H":0,"D":1,"A":3}))
        .rename(columns={"AwayTeam":"Team","AwayTarget":"Target"})
    )

    hist = pd.concat([home, away]).sort_values(["MatchDate","Team"])

    hist["SOT5"] = hist.groupby("Team")["Target"].transform(
        lambda x: x.shift(1).rolling(5).sum()
    )
    hist["Form5"] = hist.groupby("Team")["Points"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    hist = hist.dropna(subset=["Form5","SOT5"])[["MatchDate","Team","SOT5","Form5"]]

    # ensure datetime
    hist["MatchDate"] = pd.to_datetime(hist["MatchDate"])
    inputs = inputs.copy()
    inputs["MatchDate"] = pd.to_datetime(inputs["MatchDate"])

    # merge home
    df = pd.merge_asof(
        inputs.sort_values(by="MatchDate"),
        hist.sort_values(by="MatchDate"),
        left_on="MatchDate", right_on="MatchDate",
        left_by="HomeTeam", right_by="Team",
        direction="backward"
    ).rename(columns={"SOT5":"SOT5Home","Form5":"Form5Home"})

    # merge away
    df = pd.merge_asof(
        df.sort_values(by="MatchDate"),
        hist.sort_values(by="MatchDate"),
        left_on="MatchDate", right_on="MatchDate",
        left_by="AwayTeam", right_by="Team",
        direction="backward"
    ).rename(columns={"SOT5":"SOT5Away","Form5":"Form5Away"})

    # features
    df["DiffElo"]   = df["HomeElo"]  - df["AwayElo"]
    df["DiffSOT5"]  = df["SOT5Home"] - df["SOT5Away"]
    df["DiffForm5"] = df["Form5Home"] - df["Form5Away"]

    return df[["MatchDate","HomeTeam","AwayTeam","DiffElo","DiffSOT5","DiffForm5"]].reset_index(drop=True)


def predict_matches(raw_df):
    
    model = joblib.load("model_lr.pkl")
    features = build_features(raw_df)

    X = features[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
    probs = model.predict_proba(X)[:,1]

    features = features.assign(HomeWinProb = probs)
    return features[["MatchDate","HomeTeam","AwayTeam","HomeWinProb"]]

if __name__ == "__main__":
    
    matches = pd.read_csv("data/Matches.csv",low_memory=False)
    teams = pd.unique(pd.concat([matches["HomeTeam"],matches["AwayTeam"]]))

    choice = int(input("Do you want to input manual or through CSV? 1. manual, 2. CSV: "))
    
    if choice == 1:
        path = input("Please give the file name which is stored under the data/ folder: ")
        path = "data/" + path + ".csv"
        inputs = pd.read_csv(path, low_memory=False)

    elif choice == 2: 
        homeTeam, homeElo = input("Please input the home team and their ELO (team_name elo_rating): ").split()
        matches = process.extract(homeTeam, teams, limit=3)

        while matches[0][1] != 100:
            print("We found the corresponding matches")
            print(matches)
            homeTeam = input("Please input the home team again: ")
        
        awayTeam, awayElo = input("Please input the away team and their ELO (team_name elo_rating): ").split()
        matches = process.extract(awayTeam, teams, limit=3)

        while matches[0][1] != 100:
            print("We found the corresponding matches")
            print(matches)
            awayTeam = input("Please input the away team again: ")

        date = input("What is the Match Date (YYYY-MM-DD): ")

        inputs = pd.DataFrame([{
            "HomeTeam" : homeTeam,
            "AwayTeam" : awayTeam,
            "HomeElo" : int(homeElo),
            "AwayElo" : int(awayElo),
            "MatchDate" : pd.to_datetime(date)
        }])

    inputs = process_input(inputs)
    
    preds = predict_matches(inputs)    
    
    print(preds)


