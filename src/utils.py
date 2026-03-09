import pandas as pd
import os
from thefuzz import process
from datetime import datetime


def load_matches_data(file_path="data/Matches.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}\n"
                                f"Please ensure that the Matches.csv file is in the data/ folder.\n"
                                f"The file can be obtained from https://www.football-data.co.uk/englandm.php")
    
    try: 
        matches = pd.read_csv(file_path)
        return matches
    except Exception as e:
        raise ValueError(f"Failed to read matches file: {e}")
    


def get_available_teams(matches_df:pd.DataFrame, division="E0"):

    division = matches_df[matches_df["Division"]==division]

    home_teams = set(division["HomeTeam"].unique())
    away_teams = set(division["AwayTeam"].unique())
    all_teams = home_teams.union(away_teams)

    return sorted(list(all_teams))

def match_team_name(user_input, available_teams, threshold=80):
    if not available_teams:
        return None, 0
    
    res = process.extract(user_input, available_teams, limit=1)

    if not res:
        return None, 0
    
    matched_name, matched_score = res[0]  # extract() returns a list, get first tuple

    if matched_score >= threshold:
        return matched_name, matched_score
    else:
        return None, matched_score


def process_input(inputs_df, matches_df):
    home = (
        matches_df[["MatchDate","HomeTeam","HomeTarget","FTResult"]]
        .assign(Points=lambda x: x["FTResult"].map({"H":3,"D":1,"A":0}))
        .rename(columns={"HomeTeam":"Team","HomeTarget":"Target"})
    )

    away = (
        matches_df[["MatchDate","AwayTeam","AwayTarget","FTResult"]]
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
    inputs = inputs_df.copy()
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


#testing functions

if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    # Test 1: Load data
    try:
        df = load_matches_data()
        print(f"✓ Loaded {len(df)} matches")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        exit(1)
    
    # Test 2: Get teams
    teams = get_available_teams(df)
    print(f"✓ Found {len(teams)} EPL teams")
    print(f"  Sample teams: {teams[:5]}")
    
    # Test 3: Fuzzy matching
    test_inputs = ["arsenal", "mancity", "liverpool"]
    print(f"\n✓ Testing fuzzy matching:")
    for test in test_inputs:
        matched, score = match_team_name(test, teams)
        print(f"  '{test}' → '{matched}' ({score}%)")

    #Test 4: Input df creating
    
    inputs = pd.DataFrame([{
    "HomeTeam" : 'Arsenal',
    "AwayTeam" : 'Man City',
    "HomeElo" : int(2000),
    "AwayElo" : int(1900),
    "MatchDate" : pd.to_datetime("2025-01-01")
    }])

    input_df = process_input(inputs,df)
    print(pd.DataFrame.head(input_df))
    