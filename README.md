
# Binary Classification with XGBoost & Optuna Hyperparameter Tuning

This repository contains my solution for the **Kaggle Playground Series - Season 5, Episode 8** competition.  
The objective of the competition is to build a **Binary Classification Model** that predicts the target variable `y` from a given set of features.

I implemented a pipeline with:
- Data preprocessing (encoding categorical features & scaling numerical features)  
- Exploratory Data Analysis (EDA)  
- Model training using **XGBoost**  
- Cross-validation with **Stratified K-Fold**  
- Hyperparameter optimization with **Optuna**  
- Final prediction and Kaggle submission 

## Tech Stack
- **Python**
- **Pandas / NumPy** → data manipulation  
- **Matplotlib / Seaborn** → visualization  
- **Scikit-learn** → preprocessing & cross-validation  
- **XGBoost** → machine learning model  
- **Optuna** → hyperparameter tuning  

## Project Overflow
1. **Data Loading**
   - Read train and test datasets using `pandas`.

2. **Preprocessing**
   - Separate features (`X`) and target (`y`).
   - Encode categorical variables using **One-Hot Encoding** (`pd.get_dummies`).
   - Align training and test feature sets to ensure consistency.
   - Scale numerical features with **StandardScaler**.

3. **Cross-Validation**
   - Used **Stratified K-Fold** to evaluate model robustness.
   - Evaluated using **ROC-AUC score**.

4. **Hyperparameter Tuning**
   - Used **Optuna** to optimize XGBoost parameters (`max_depth`, `learning_rate`, `subsample`, etc.).
   - Objective function maximized the mean **CV AUC score**.

5. **Model Training**
   - Trained final **XGBoost Classifier** with best parameters.

6. **Prediction**
   - Generated probabilities (`predict_proba`) for the test dataset.

7. **Submission**
   - Created a submission file (`submission.csv`) with `id` and predicted `y`.

## Results
- **Cross-validation AUC:** ~0.9664 ± 0.0005  
- Model demonstrated **high consistency across folds**.  
- Final submission was prepared in Kaggle format. 

## How to run 
1. Clone this repository:
   ```bash
   git clone https://github.com/SahilChukka19/Binary-Classification-With-Banking-Dataset.git
   ```
2. Install dependencies:
   ```bash
   pip install requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```
## Resources
- [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/overview)
- [My Kaggle Submission](https://www.kaggle.com/code/sahilchukka19/binary-classification-bank-dataset)
- [My Medium Blog](https://medium.com/@sahil.chukka/binary-classification-on-banking-dataset-a-kaggle-competition-9848816a387d)




