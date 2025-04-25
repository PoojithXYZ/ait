import joblib
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import polars as pl
import os

from cat_class import My_Cat_B_Model
from xgboost_class import My_XGB_Model
from light_class import My_Light_Model


class Ensemble:
    def __init__(self):
        self.xgb_model = My_XGB_Model()
        self.lgbm_model = My_Light_Model()
        self.catboost_model = My_Cat_B_Model()
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip') # Or other strategies
    
    def weighted_avg(self, X, weights=None):
        xgb_predictions = self.xgb_model.predict(X)
        lgbm_predictions = self.lgbm_model.predict(X)
        catboost_predictions = self.catboost_model.predict(X)
        return np.average([xgb_predictions, lgbm_predictions, catboost_predictions], axis=0, weights=weights)

    def fit_iso(self, X, y):
        xgb_preds = self.xgb_model.predict(X_train)
        lgbm_preds = self.lgbm_model.predict(X_train)
        catboost_preds = self.catboost_model.predict(X_train)
        meta_feature = np.mean([xgb_preds, lgbm_preds, catboost_preds], axis=0)

        self.isotonic_regressor.fit(meta_feature, y_train)
        joblib.dump(self.isotonic_regressor, 'iso_regressor.joblib')

    def predict_iso(self, X):
        self.isotonic_regressor = joblib.load('iso_regressor.joblib')
        xgb_predictions = self.xgb_model.predict(X)
        lgbm_predictions = self.lgbm_model.predict(X)
        catboost_predictions = self.catboost_model.predict(X)

        meta_feature = np.mean([xgb_predictions, lgbm_predictions, catboost_predictions], axis=0)
        return self.isotonic_regressor.predict(meta_feature)


if __name__ == "__main__":

    path_to_data = './data' # <--- **CHANGE THIS TO YOUR DATA DIRECTORY**
    train_file = 'train.csv' # <--- **CHANGE THIS TO YOUR TRAINING DATA FILE**
    test_file = 'test.csv' # <--- **CHANGE THIS TO YOUR TESTING DATA FILE**
    full_train_path = os.path.join(path_to_data, train_file)
    full_test_path = os.path.join(path_to_data, test_file)

    print(f"Loading data from: {full_train_path}")
    df = pl.read_csv(full_train_path).to_pandas()
    print("Data loaded successfully.")

    cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', 'utility_agent1', 'Id'] # Added 'Id' for consistency
    X = df.drop(columns=cols_to_drop, axis=1, errors='ignore')
    y = df['utility_agent1']

    # Perform any necessary preprocessing steps here (e.g., encoding, scaling)
    # For demonstration, we'll assume X is ready.
    # <--- **ADD YOUR PREPROCESSING CODE HERE IF NEEDED TO GET X_encoded**
    # X # Placeholder: assuming no encoding is needed or it's already done


    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    ensemble = Ensemble()

    ensemble_predictions_avg = ensemble.weighted_avg(X_train, weights=None)

    print("\nWeighted Average Predictions (first 5):", ensemble_predictions_avg[:5])
    ensemble_rmse_avg = np.sqrt(mean_squared_error(y_test, ensemble_predictions_avg))
    print("Weighted Average RMSE:", ensemble_rmse_avg)


    ensemble.fit_iso(X_train, y_train)
    ensemble_predictions_iso = ensemble.predict_iso(X_test)
    print("\nIsotonic Ensemble Predictions (first 5):", ensemble_predictions_iso[:5])
    ensemble_rmse_iso = np.sqrt(mean_squared_error(y_test, ensemble_predictions_iso))
    print("Isotonic Ensemble RMSE:", ensemble_rmse_iso)

