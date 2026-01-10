import joblib
import pandas as pd
from features import build_features
from thefuzz import fuzz, process

def input_prep(input_raw):
    
    

    return inputs

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
            "MatchDate" : pd.to_datetime(pd.series(date))
        }])

    inputs = input_prep(inputs)
    
    preds = predict_matches(inputs)    
    
    print(preds)



    '''
    for tomorrow, you can try to someway to make the inputs file to have the DiffSOT5 and the DiffForm5
    by maybe redesigning the build_features or implement the whole new thing
    for redesigning the build_features maybe you will have to sort, and then search for the matches before the date only, so make sure that you have the right approach
    '''
