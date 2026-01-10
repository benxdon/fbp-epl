import pandas as pd

def rolling_mean_prev_k(df, k):
    shifted = df.shift(1)
    return shifted.rolling(k).mean()

def build_features(matches):
    matches = matches[matches["Division"] == "E0"].copy()

    # build team-level table
    home = matches[["MatchDate","HomeTeam","HomeTarget"]].rename(
        columns={"HomeTeam":"Team", "HomeTarget":"SOT"}
    )
    away = matches[["MatchDate","AwayTeam","AwayTarget"]].rename(
        columns={"AwayTeam":"Team", "AwayTarget":"SOT"}
    )

    team_matches = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["Team","MatchDate"])
    )

    # rolling SOT
    team_matches["SOT5"] = (
        team_matches
        .groupby("Team")["SOT"]
        .transform(lambda s: rolling_mean_prev_k(s, 5))
    )

    # merge back
    matches = matches.merge(
        team_matches[["MatchDate","Team","SOT5"]],
        left_on=["MatchDate","HomeTeam"],
        right_on=["MatchDate","Team"],
        how="left"
    ).rename(columns={"SOT5":"HomeSOT5"}).drop(columns="Team")

    matches = matches.merge(
        team_matches[["MatchDate","Team","SOT5"]],
        left_on=["MatchDate","AwayTeam"],
        right_on=["MatchDate","Team"],
        how="left"
    ).rename(columns={"SOT5":"AwaySOT5"}).drop(columns="Team")

    # final features
    features = (
        matches
        .assign(
            DiffElo = matches["HomeElo"] - matches["AwayElo"],
            Result = (matches["FTResult"]=="H").astype(int),
            HomeAdv = 1,
            DiffForm5 = matches["Form5Home"] - matches["Form5Away"],
            DiffSOT5 = matches["HomeSOT5"] - matches["AwaySOT5"]
        )
        [["MatchDate","DiffElo","DiffForm5","HomeAdv","DiffSOT5","Result"]]
        .dropna()
    )

    return features
