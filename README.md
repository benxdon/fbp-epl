#EPL Home Win Probability Predictor

A small CLI tool that estimates the probability of a **home team win** in EPL matches.
No crystal balls, no guarantees. Just math, history, and a little football logic.

It uses:

* Elo rating difference
* Recent form (last 5 matches)
* Shots on target (last 5 matches)
* A simple logistic regression model

The output is a probability, not a win/loss verdict.
Think *“how likely”*, not *“who will win.”*

---

## Getting Started

First, clone the repository:

```bash
git clone https://github.com/benxdon/epl-football-match-prediction
cd ml_football
```

Make sure you have a `Matches.csv` file inside the `data/` folder.
This file is used to build historical features and train the model.

You can get the data from:

* [https://github.com/xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
* [https://www.football-data.co.uk/](https://www.football-data.co.uk/)

---

## Train the Model

Before predicting anything, train the model:

```bash
python src/train.py
```

This builds the logistic regression model using historical EPL data.

---

## Run Predictions

To make predictions, you will need:

* Home team Elo
* Away team Elo

You can find Elo ratings at:

* [http://clubelo.com/](http://clubelo.com/)

Then run:

```bash
python src/predict.py
```

Follow the CLI prompts:

* Choose manual or CSV input
* Enter teams and Elo ratings
* Confirm fuzzy-matched team names
* Receive a home win probability

Example output:

```
HomeWinProb = 0.63
```

Interpretation:
The model estimates a 63% chance that the home team wins.

---

## Naming Convention

When entering team names:

* Use one word only
* Examples:

  * `mancity` instead of `Man City`
  * `manutd` instead of `Man United`

The CLI will help confirm the correct team name using fuzzy matching.

---

## Want to Go Deeper?

To understand how the data is transformed:

* Check the `transform.py` file in the `notebooks/` folder
* Explore the R notebooks that analyze how Elo differences affect win probability

That’s where the “why” behind the model lives.

---

## Disclaimer

* This project is for learning and analysis only
* Not for betting or financial decisions
* Football is chaotic, models are humble
* Use probabilities responsibly
