import joblib
import pandas as pd
import os
from utils import load_matches_data, process_input, get_available_teams, match_team_name 

def predict_matches(inputs, model_name="model_lr"):
    
    model_path = f"models/{model_name}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first by running 'python3 src/train.py'"
        )
    
    model = joblib.load(model_path)

    X = inputs[["DiffElo","DiffForm5","HomeAdv","DiffSOT5"]]
    probs = model.predict_proba(X)[:,1]

    inputs = inputs.assign(HomeWinProb = probs)
    return inputs[["MatchDate","HomeTeam","AwayTeam","HomeWinProb"]]

if __name__ == "__main__":
    
    print("=" * 60)
    print("EPL MATCH PREDICTION")
    print("=" * 60)
    print("\nWhere to find Elo ratings:")
    print("  → https://footballdatabase.com/")
    print("  → https://www.football-data.co.uk/englandm.php")
    print("=" * 60)
    
    try:
        matches = load_matches_data()
        teams = get_available_teams(matches)
        print(f"\n✓ Loaded {len(matches)} matches")
        print(f"✓ Found {len(teams)} EPL teams\n")
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        exit(1)

    choice = int(input("Do you want to input manual or through CSV?\n  1. CSV\n  2. Manual\nChoice: "))
    
    if choice == 1:
        path = input("\nPlease give the file name which is stored under the data/ folder: ")
        path = "data/" + path + ".csv"
        inputs = pd.read_csv(path, low_memory=False)

    elif choice == 2: 
        print("\n" + "-" * 60)
        print("MANUAL INPUT MODE")
        print("-" * 60)

        while True:
            print("\n[HOME TEAM]")
            homeTeam_input = input("Enter home team name: ").strip()
            homeElo_input = input("Enter home team Elo rating: ").strip()
            
            # Use shared fuzzy matching function
            matched_home, score = match_team_name(homeTeam_input, teams)
            
            if matched_home is None:
                print(f"✗ No good match found for '{homeTeam_input}'. Try again.")
                continue
            
            print(f"✓ Best match: {matched_home} ({score}% confidence)")
            confirm = input("Use this team? (y/n): ").lower()

            if confirm == "y":
                homeTeam = matched_home
                homeElo = int(homeElo_input)
                break
            
        while True:
            print("\n[AWAY TEAM]")
            awayTeam_input = input("Enter away team name: ").strip()
            awayElo_input = input("Enter away team Elo rating: ").strip()
            
            # Use shared fuzzy matching function  
            matched_away, score = match_team_name(awayTeam_input, teams)
            
            if matched_away is None:
                print(f"✗ No good match found for '{awayTeam_input}'. Try again.")
                continue
            
            print(f"✓ Best match: {matched_away} ({score}% confidence)")
            confirm = input("Use this team? (y/n): ").lower()

            if confirm == "y":
                awayTeam = matched_away
                awayElo = int(awayElo_input)
                break

        date = input("\nWhat is the Match Date (YYYY-MM-DD): ")

        inputs = pd.DataFrame([{
            "HomeTeam" : homeTeam,
            "AwayTeam" : awayTeam,
            "HomeElo" : homeElo,
            "AwayElo" : awayElo,
            "MatchDate" : pd.to_datetime(date)
        }])

    print("\n" + "=" * 60)
    print("CALCULATING PREDICTION...")
    print("=" * 60)
    
    inputs = process_input(inputs,matches)
    preds = predict_matches(inputs)    
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(preds)
    print("=" * 60)



