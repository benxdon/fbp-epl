import pandas as pd 

#this file is to help understand how inputs can be transformed into the correct table format 

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

#as you can see, the hist table contains the information about the prev matches
print(hist)


#in the inputs, we need to get the DiffElo, DiffForm5, and DiffSOT5. therefore we need to do a merge to get the data
inputs = pd.DataFrame([{
    "HomeTeam":"Paris SG",
    "HomeElo":1300,
    "AwayTeam":"Bordeaux",
    "AwayElo":1000,
    "MatchDate":"2008-01-01"
    }])

inputs["MatchDate"] = pd.to_datetime(inputs["MatchDate"])
hist["MatchDate"] = pd.to_datetime(hist["MatchDate"])


#merge process
processed = pd.merge_asof(
        inputs.sort_values("MatchDate"),
        hist.sort_values("MatchDate"),
        left_on = "MatchDate",
        right_on = "MatchDate",
        left_by = "HomeTeam",
        right_by = "Team", 
        direction = "backward"
        ).rename(columns={"SOT5":"SOT5Home","Form5":"Form5Home"})


processed = pd.merge_asof(
        processed.sort_values("MatchDate"),
        hist.sort_values("MatchDate"),
        left_on = "MatchDate",
        right_on = "MatchDate",
        left_by = "AwayTeam",
        right_by = "Team", 
        direction = "backward"
        ).rename(columns={"SOT5":"SOT5Away","Form5":"Form5Away"})

processed = (
        processed
        .assign(
            DiffElo = lambda x: x["HomeElo"] - x["AwayElo"],
            DiffSOT5 = lambda x: x["SOT5Home"] - x["SOT5Away"],
            DiffForm5 = lambda x: x["Form5Home"] - x["Form5Away"])
        .loc[:, ["MatchDate","HomeTeam","AwayTeam","DiffElo","DiffSOT5","DiffForm5"]]
        )

print(processed)
