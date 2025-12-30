**TL;DR:** A probability-based model using Elo, recent form, and shot statistics to estimate EPL home win likelihoods.

# ⚽ EPL Match Outcome Prediction

This project builds a **probability-based machine learning model** to estimate the likelihood of a **home team win** in English Premier League (EPL) matches using **pre-match information only**.

The project is structured in two main stages:

1. **Exploratory Data Analysis (R)** — understanding Elo and outcome relationships  
2. **Machine Learning Pipeline (Python)** — feature engineering, modeling, and prediction

The focus is on **probability estimation**, not deterministic win/loss prediction.

---

## 📌 Project Motivation

Football match outcomes are inherently uncertain.  
Rather than predicting *who will win*, this project aims to answer:

> *“How likely is a home team win, given what we know before kickoff?”*

This probabilistic framing:
- Better reflects real-world uncertainty
- Enables proper evaluation with log loss
- Produces outputs that are interpretable and reusable

---

## 🧪 Stage 1: Exploratory Data Analysis (R)

Before building any machine learning model, an exploratory analysis was conducted in **R** to understand whether **Elo rating differences** contain predictive signal.

### Goals of the R Analysis
- Validate Elo difference as a meaningful feature
- Examine the relationship between Elo difference and win probability
- Motivate the use of a probabilistic classification model

### Key Steps
- Filtered EPL matches
- Computed `DiffElo = HomeElo − AwayElo`
- Binned Elo differences
- Calculated empirical home win rates per bin
- Visualized the monotonic relationship between Elo difference and win probability

### Conclusion from R Analysis
- Home win probability increases monotonically with Elo difference
- Elo difference is a strong baseline feature
- The relationship is approximately linear in log-odds

📌 This analysis motivated:
- Binary classification (home win vs not)
- Logistic regression as a natural baseline model

---

## 🧠 Stage 2: Machine Learning Pipeline (Python)

After validating feature relevance in R, the full modeling pipeline was implemented in **Python**.

### Target
Binary classification:
- `1` → Home team wins  
- `0` → Home team does not win

### Features
All features are computed using **only information available before the match**.

| Feature | Description |
|------|------------|
| `DiffElo` | Home Elo − Away Elo |
| `DiffForm5` | Difference in last 5-match form |
| `DiffSOT5` | Difference in average shots on target over previous 5 matches |
| `HomeAdv` | Constant home advantage indicator |

Rolling features use **shifted windows** to prevent data leakage.

---

## 📊 Models Tested

Two models were implemented and compared using the **same features and time-based split**.

### 1️⃣ Logistic Regression (Final Model)
- Interpretable
- Well-calibrated probabilities
- Best log loss and AUC

### 2️⃣ Random Forest
- Captures non-linearities
- Slightly higher accuracy
- Worse probability calibration

📌 **Logistic Regression was retained as the final model** due to superior probabilistic performance.

---

## 📈 Evaluation Metrics

Primary metric:
- **Log loss** (probability quality)

Secondary metric:
- **ROC AUC** (ranking ability)

Accuracy is reported for reference only.

### Final Results

| Model | Log Loss | AUC | Accuracy |
|----|----|----|----|
| Logistic Regression | ~0.609 | ~0.720 | ~0.66 |
| Random Forest | ~0.611 | ~0.719 | ~0.67 |

---

## 🗂️ Project Structure
ml_football/
├── data/ # (not committed) raw CSV data
├── notebooks/ # R analysis / exploratory work
│ └── EPL_EDA.Rmd
├── src/
│ ├── features.py # feature engineering
│ ├── train.py # model training & evaluation
│ └── predict.py # prediction interface
├── README.md
├── .gitignore


## How to run

### R Analysis (optional)
Open the R Markdown file in `notebooks/` to reproduce the exploratory analysis.


### Install the Python dependencies

```powershell
pip install pandas numpy scikit-learn joblib
```

### Train the model
```powershell
python src\train.py
```
This trains the model and reports evaluation metrics

### Generate predictions
```powershell
python src\predict.py
```
You will be prompted to enter the name of a CSV file located in the `data/` folder

## Outputs
The prediction script output home win probabilities, not class labels
For example: `HomeWinProb = 0.63`

Interpretaion: Based on pre-match information, the model estimates a 63% chance that the home team wins


## Limitations
- No injuries, or in-game events
- Assumes historical patterns remain stable
- EPL only
- Not intended for betting or financial decision-making
This project is for educational and analytical purposes


## 📚 Data Sources

This project uses publicly available football match data.

### Match Results & Match Statistics
- Source: **Football-Data.co.uk**
- Website: https://www.football-data.co.uk/
- Data includes:
  - Match results
  - Goals scored
  - Shots on target
  - Match dates
  - League identifiers
  - Betting odds (not used for modeling)

The dataset is provided freely for personal and educational use.  
Due to licensing and file size considerations, raw data files are **not included** in this repository.

### Elo Ratings
- Elo ratings were sourced from publicly available football Elo rating datasets.
- Elo ratings represent long-term team strength and are updated based on historical match outcomes.

---

## 📌 Data Usage Notes

- Only **pre-match information** is used for feature construction.
- No in-game or post-match variables are used in prediction.
- All rolling features are computed using past matches only to prevent data leakage.

This project is for **educational and analytical purposes**.

