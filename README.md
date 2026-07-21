# Medical Insurance Cost Prediction — Multiple Linear Regression

## Objective
Build a Multiple Linear Regression model to predict medical insurance `charges`
for customers based on personal and health-related information (age, sex, BMI,
number of children, smoking status, and region).

## Dataset Link
[Medical Cost Personal Insurance Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

> **Note:** The dataset is not included in this repository. Download `insurance.csv`
> from the Kaggle link above and place it in a local `data/` folder before running
> the notebook (see Methodology below).

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, Linear Regression model, evaluation metrics
- `matplotlib` — Actual vs Predicted visualization
- `jupyter` — notebook environment

## Methodology
1. **Data Understanding** — Loaded the dataset, previewed records, and identified
   numerical features (`age`, `bmi`, `children`), categorical features (`sex`,
   `smoker`, `region`), and the target variable (`charges`).
2. **Data Preprocessing** — Checked for missing values (none found), label-encoded
   `sex` and `smoker` (binary), one-hot encoded `region` (`drop_first=True` to
   avoid the dummy-variable trap), and split the data 80/20 into train/test sets
   (`random_state=42`).
3. **Model Development** — Trained a `LinearRegression` model from scikit-learn
   on the encoded training features, then generated predictions on the held-out
   test set.
4. **Model Evaluation** — Scored predictions using MAE, MSE, and R², and plotted
   Actual vs Predicted charges to visually assess fit.

## Results
| Metric | Value |
|--------|-------|
| MAE    | 4181.19 |
| MSE    | 33,596,915.85 |
| RMSE   | 5796.28 |
| R²     | 0.7836 |

**Observations:**
1. The model explains ~78% of the variance in charges (R² = 0.78), a solid fit
   for a simple linear model, though a meaningful share of variance remains
   unexplained.
2. Predictions are noticeably tighter for non-smokers with lower charges, while
   smokers and high-charge outliers show much larger residuals — the true
   relationship between smoking and charges appears non-linear (a steep jump
   rather than a proportional increase).
3. `smoker` has by far the largest coefficient magnitude, confirming it is the
   dominant driver of charges, followed by `age` and `bmi`; `sex` and `region`
   contribute comparatively little.

## Conclusion
This Multiple Linear Regression model predicts medical insurance charges using
age, sex, BMI, number of children, smoking status, and region. Smoking status
emerged as the strongest predictor, with smokers incurring substantially higher
charges than non-smokers, followed by age and BMI, both of which increase
charges as they rise. Sex and region had comparatively minor effects. The model
achieved a reasonably strong R² score on the test set, showing it captures the
broad cost drivers well, though prediction errors were larger for smokers and
high-BMI individuals, whose charges rise steeply rather than steadily.

**Limitation:** Linear Regression assumes a linear, additive relationship between
features and charges, but the real relationship — particularly the sharp jump in
cost for smokers combined with high BMI — is non-linear and interactive. This
means the model likely underfits these high-cost segments, and a technique
capable of capturing feature interactions (e.g., polynomial regression, decision
trees, or gradient boosting) could better model this data.

## Project Structure
```
Assignment-1/
├── Assignment-1.ipynb   # Tasks 1-5, one section per task
├── README.md
├── requirements.txt
└── .gitignore           # excludes data/ (dataset not redistributed)
```

## How to Run
```bash
pip install -r requirements.txt
# Download insurance.csv from the Kaggle link above into a local data/ folder
jupyter notebook Assignment-1.ipynb
```