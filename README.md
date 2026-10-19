# Titanic: Predicting Passenger Survival

A machine learning project on Kaggle's [Titanic competition](https://www.kaggle.com/competitions/titanic): predicting which passengers survived the sinking from details such as sex, age, class and fare.

## Approach

1. **Data quality checks:** shapes, data types, duplicates and missing values in both the training and test files.
2. **Cleaning:**
   - Missing ages filled with the median age for each passenger's title (Mr, Mrs, Miss, Master).
   - Missing ports filled with the most common port.
   - The mostly empty cabin column turned into a "has cabin" flag and a deck letter.
3. **Exploratory analysis:** survival rates by sex, class, title, port, age and deck, including combinations such as sex with class.
4. **Feature engineering:** title, family size, travelling alone, fare per person and log fare.
5. **Modelling:** logistic regression, random forest and XGBoost, built as scikit-learn pipelines and compared against two baselines.
6. **Evaluation:** stratified 5-fold cross-validation, a check for overfitting, hyperparameter tuning with grid search, and precision, recall and F1 scores.

## Results

Cross-validated accuracy on the training data:

| Model | Accuracy |
|---|---|
| Majority baseline (everyone dies) | 0.62 |
| Gender rule (women survive, men die) | 0.79 |
| Logistic regression (tuned) | 0.83 |
| Random forest (tuned) | 0.84 |
| XGBoost (tuned) | 0.85 |

The tuned scores are the best scores from grid search, which tend to be slightly optimistic. The three models are within about two points of each other.

**Key findings**

- Sex was the strongest predictor: 74% of women survived against 19% of men.
- Class mattered differently for each sex. Women in third class (50%) still outlived men in first class (37%).
- The untuned random forest scored 99% on its training data against 80% on validation, a clear sign of overfitting.

## Project status

Work in progress. Next steps:

- Fit the final model on all the training data.
- Predict the test set and submit to Kaggle.
- Record the leaderboard score here.

## How to run

1. Clone this repository and create a virtual environment:
```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
```
2. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the project folder. The data is not included, in line with Kaggle's rules.
3. Open `titanic_notebook.ipynb` in JupyterLab and run all cells.

## Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, JupyterLab.

## Author

Matthew Ilori