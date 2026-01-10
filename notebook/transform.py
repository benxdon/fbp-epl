import pandas as pd 

#import the past data
matches = pd.read_csv("data/Matches.csv", low_memory=False)

#create a table for home team
home_hist = matches[["MatchDate","HomeTeam", "HomeTarget","FTResult"]]

home_hist = (
        home_hist
        .assign(Points = lambda df: df["FTResult"].map({"H":3, "D":1, "A":0}))
        .rename(columns={
            "HomeTeam":"Team",
            "HomeTarget":"Target"
            })
        )

#same with away 
away_hist = matches[["MatchDate","AwayTeam","AwayTarget","FTResult"]]

away_hist = (
        away_hist
        .assign(Points = lambda df: df["FTResult"].map({"H":0, "D":1, "A":3}))
        .rename(columns={
            "AwayTeam":"Team",
            "AwayTarget":"Target"
            })
        )

#hist will be the table containing past data
hist = pd.concat([home_hist, away_hist]).sort_values(["MatchDate","Team"])

#from the hist table, we will need to calculate the rolling average and rolling sum for the last 5 matches 
hist["SOT5"] = (
        hist
        .groupby("Team")["Target"]
        .transform(lambda x: x.shift(1).rolling(5).sum())
        )

hist["Form5"] = (
        hist
        .groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).rolling(5).mean())
        )

hist = hist.dropna(subset = ["Form5","SOT5"]).drop(columns = ["FTResult","Points","Target"])

print(hist)

inputs = pd.DataFrame([{
    "HomeTeam":"Paris SG",
    "HomeElo":1300,
    "AwayTeam":"Bordeaux",
    "AwayElo":1000,
    "MatchDate":"2008-01-01"
    }])


