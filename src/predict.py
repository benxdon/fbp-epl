import joblib
import pandas as pd
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


    #remove the values which have NA
    df = df.dropna(subset=["SOT5Home","SOT5Away","Form5Home","Form5Away"])

    # features
    df["DiffElo"]   = df["HomeElo"]  - df["AwayElo"]
    df["DiffSOT5"]  = df["SOT5Home"] - df["SOT5Away"]
    df["DiffForm5"] = df["Form5Home"] - df["Form5Away"]
    df["HomeAdv"] = 1

    return df[["MatchDate","HomeTeam","AwayTeam","DiffElo","DiffSOT5","DiffForm5","HomeAdv"]].reset_index(drop=True)


def predict_matches(inputs):
    
    model = joblib.load("model_lr.pkl")

    X = inputs[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
    probs = model.predict_proba(X)[:,1]

    inputs = inputs.assign(HomeWinProb = probs)
    return inputs[["MatchDate","HomeTeam","AwayTeam","HomeWinProb"]]

if __name__ == "__main__":
    
    matches = pd.read_csv("data/Matches.csv",low_memory=False)
    teams = pd.unique(pd.concat([matches["HomeTeam"],matches["AwayTeam"]]))

    choice = int(input("Do you want to input manual or through CSV? 1. CSV, 2. manual: "))
    
    if choice == 1:
        path = input("Please give the file name which is stored under the data/ folder: ")
        path = "data/" + path + ".csv"
        inputs = pd.read_csv(path, low_memory=False)

    elif choice == 2: 

        while True:
            
            homeTeam, homeElo = input("Please input the home team and their ELO (team_name elo_rating): ").split()
            matches = process.extract(homeTeam, teams, limit=3)
            
            best_team, best_score = matches[0]

            print(f"Best match: {best_team} ({best_score}%)")
            confirm = input("Use this team? (y/n): ").lower()

            if confirm == "y":
                homeTeam = best_team
                break

            print("Other possible matches: ")
            for i, (team, score) in enumerate(matches):
                print(f"{i+1}. {team} ({score})")

            choice = int(input("Choose 1-3: "))
            
            if choice >= 1 and choice <= 3:
                homeTeam = matches[choice-1][0]
                break
            
        while True:
            
            awayTeam, awayElo = input("Please input the away team and their ELO (team_name elo_rating): ").split()
            matches = process.extract(awayTeam, teams, limit=3)
            
            best_team, best_score = matches[0]

            print(f"Best match: {best_team} ({best_score}%)")
            confirm = input("Use this team? (y/n): ").lower()

            if confirm == "y":
                awayTeam = best_team
                break

            print("Other possible matches: ")
            for i, (team, score) in enumerate(matches):
                print(f"{i+1}. {team} ({score})")

            choice = int(input("Choose 1-3: "))
            
            if choice >= 1 and choice <= 3:
                awayTeam = matches[choice-1][0]
                break


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


