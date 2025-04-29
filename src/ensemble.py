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
    self.target = 'utility_agent1'
    self.base_preds = []

    def __init__(self):
        self.xgb_model = My_XGB_Model()
        self.lgbm_model = My_Light_Model()
        self.catboost_model = My_Cat_B_Model()
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    
    def run_base_models(self, X):
        xgb_predictions = self.xgb_model.model.predict(X)
        lgbm_predictions = self.lgbm_model.predict(X)
        catboost_predictions = self.catboost_model.predict(X)

        self.base_preds = [xgb_predictions, lgbm_predictions, catboost_predictions]
        # return self.base_preds

    def weighted_avg(self, X, weights=None):
        bp = self.base_preds
        return np.average([bp[0], bp[1], bp[2]], axis=0, weights=weights)

    def fit_iso(self, X, y):
        self.isotonic_regressor.fit(meta_fea, y_train)
        joblib.dump(self.isotonic_regressor, 'iso_regressor.joblib')

    def predict_iso(self, X):
        self.isotonic_regressor = joblib.load('iso_regressor.joblib')
        iso_preds = self.isotonic_regressor.predict(meta_feature)
        return iso_preds


if __name__ == "__main__":

    path_to_data = './data'
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_path = os.path.join(path_to_data, train_file)
    test_path = os.path.join(path_to_data, test_file)

    print(f"Loading data from: {train_path}")
    df = pd.read_csv(train_path)
    print("Data loaded successfully.")

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

