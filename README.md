# EPL Home Win Probability Predictor

A machine learning CLI tool that estimates the probability of a **home team win** in EPL matches.
No crystal balls, no guarantees. Just math, history, and a little football logic.

## Features

- **Multiple ML Models**: Compare predictions from Logistic Regression, Random Forest, and XGBoost
- **Smart Team Matching**: Fuzzy string matching handles typos (e.g., "mancity" → "Man City")
- **Historical Context**: Analyzes 230,000+ historical matches dating back to 2000
- **Feature Engineering**: Combines Elo ratings, recent form, shots on target, and home advantage
- **Interactive CLI**: User-friendly prompts guide you through predictions

## How It Works

The model considers:

* **Elo rating difference** - Team strength indicator
* **Recent form** (last 5 matches) - Momentum and current performance
* **Shots on target** (last 5 matches) - Attacking efficiency
* **Home advantage** - Historical home field benefit

The output is a probability, not a win/loss verdict.
Think *"how likely"*, not *"who will win."*

---

## Tech Stack

- **Python 3.12** - Core language
- **pandas 3.0.1** - Data manipulation
- **scikit-learn 1.8.0** - Machine learning (Logistic Regression, Random Forest)
- **XGBoost 3.2.0** - Gradient boosting classifier
- **thefuzz 0.22.1** - Fuzzy string matching for team names
- **joblib 1.5.3** - Model persistence

---

## Model Performance

Trained on historical EPL data (2000-2025):

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~66.5% |
| Random Forest | ~66.2% |
| XGBoost | ~66%+ |

*Note: Football is inherently unpredictable; these models provide informed probabilities, not guarantees.*

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/benxdon/fbp-epl
cd fbp-epl
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv ml_env
source ml_env/bin/activate  # On Linux/Mac
# ml_env\Scripts\activate   # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Match Data

Place a `Matches.csv` file inside the `data/` folder.

**Data Sources:**

* [Club Football Match Data (2000-2025)](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
* [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)

---

## Project Structure

```
fbp-epl/
├── data/
│   ├── Matches.csv        # Historical match data
│   └── EloRatings.csv     # (Optional) Historical Elo ratings
├── models/
│   ├── model_lr.pkl       # Trained Logistic Regression
│   ├── model_rf.pkl       # Trained Random Forest
│   └── model_xgb.pkl      # Trained XGBoost
├── src/
│   ├── features.py        # Feature engineering functions
│   ├── train.py           # Model training script
│   ├── predict.py         # Prediction CLI
│   └── utils.py           # Shared utility functions
├── notebook/
│   ├── EPL_EDA.Rmd        # Exploratory data analysis
│   └── transform.py       # Data transformation experiments
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Usage

### Step 1: Train the Models

Before making predictions, train all three models:

```bash
python3 src/train.py
```

This will:
- Load historical match data
- Engineer features (Elo differences, form, shots on target)
- Train Logistic Regression, Random Forest, and XGBoost models
- Save trained models to `models/` directory
- Display accuracy metrics for each model

Expected output:
```
Training Logistic Regression...
✓ Accuracy: 0.6647

Training Random Forest...
✓ Accuracy: 0.6622

Training XGBoost...
✓ Accuracy: 0.6610
```

---

### Step 2: Make Predictions

Run the prediction CLI:

```bash
python3 src/predict.py
```

#### Finding Elo Ratings

You'll need current Elo ratings for both teams. Get them from:

* **[Football Database](https://footballdatabase.com/)** - Most reliable, updated regularly
* **[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)** - Alternative source

---

#### Interactive Example

```
============================================================
EPL MATCH PREDICTION
============================================================

Do you want to input manual or through CSV?
  1. CSV
  2. Manual
Choice: 2

------------------------------------------------------------
MANUAL INPUT MODE
------------------------------------------------------------

[HOME TEAM]
Enter home team name: arsenal
Enter home team Elo rating: 2000
✓ Best match: Arsenal (100% confidence)
Use this team? (y/n): y

[AWAY TEAM]
Enter away team name: mancity
Enter away team Elo rating: 1900
✓ Best match: Man City (93% confidence)
Use this team? (y/n): y

What is the Match Date (YYYY-MM-DD): 2025-01-01

============================================================
CALCULATING PREDICTION...
============================================================

============================================================
RESULT:
============================================================
   MatchDate HomeTeam  AwayTeam  HomeWinProb
0 2025-01-01  Arsenal  Man City     0.596913
============================================================
```

**Interpretation**: The model estimates a ~59.7% probability that Arsenal wins at home against Man City.

---

## Team Name Tips

The CLI uses fuzzy matching, so you don't need exact names:

| You Type | Matches To |
|----------|------------|
| `arsenal` | Arsenal |
| `mancity` | Man City |
| `manutd` | Man United |
| `liverpool` | Liverpool |
| `spurs` | Tottenham |

If the match confidence is low (<80%), the CLI will ask for confirmation.

---

## Development & Analysis

To understand the feature engineering and model selection:

* **`src/features.py`** - Feature engineering logic (rolling averages, Elo differences)
* **`notebook/transform.py`** - Data transformation experiments
* **`notebook/EPL_EDA.Rmd`** - R notebooks analyzing Elo vs win probability relationships

That's where the "why" behind the model lives.

---

## Future Improvements

- [ ] Add ensemble voting across all three models
- [ ] Implement cross-validation for more robust accuracy estimates
- [ ] Add draw probability prediction (currently only home win)
- [ ] Historical backtesting mode to evaluate past predictions
- [ ] CSV batch prediction for multiple matches
- [ ] Player injury/suspension impact features
- [ ] Head-to-head historical record weighting
- [ ] Confidence intervals for predictions
- [ ] Docker containerization for easier deployment

---

## Disclaimer

* **This project is for learning and analysis only**
* **Not intended for betting or financial decisions**
* Football is inherently chaotic and unpredictable
* Models provide informed estimates, not certainties
* Past performance doesn't guarantee future results
* Use probabilities responsibly and for research purposes

---

## Contributing

This is a portfolio/learning project, but suggestions and improvements are welcome!

Feel free to:
- Open issues for bugs or feature requests
- Fork and experiment with your own features
- Share insights on model improvements

---

## License

This project is open source and available for educational purposes.

---

## Acknowledgments

- Match data from [xgabora's Club Football dataset](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
- Elo ratings methodology inspired by [Football Database](https://footballdatabase.com/)
- Built as a learning project to explore ML in sports analytics
