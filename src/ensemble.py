import joblib
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import polars as pl
import os
import csv

from cat_class import My_Cat_B_Model
from xgboost_class import My_XGB_Model
from light_class import My_Light_Model


class Ensemble:
    def __init__(self):
        self.target = 'utility_agent1'
        self.base_preds = []
        self.meta = None
        self.xgb_model = My_XGB_Model()
        self.lgbm_model = My_Light_Model()
        self.catboost_model = My_Cat_B_Model()
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    
    def run_base_models(self, data_path):
        xgb_predictions = self.xgb_model.predict(data_path)
        lgbm_predictions = self.lgbm_model.predict(data_path)
        catboost_predictions = self.catboost_model.predict(data_path)

        self.base_preds = [xgb_predictions, lgbm_predictions, catboost_predictions]
        # return self.base_preds

    def weighted_avg(self, data_path, weights=None):
        if not self.base_preds:
            self.run_base_models(data_path)
        bp = self.base_preds
        self.meta = pd.DataFrame(
            np.average([bp[0], bp[1], bp[2]], axis=0, weights=weights),
            columns=['Average_Prediction']
        )
        return self.meta

    def fit_iso(self, data_path, y):
        if not self.base_preds:
            self.run_base_models(data_path)
        self.meta = self.weighted_avg(data_path)
        self.isotonic_regressor.fit(self.meta, y)
        joblib.dump(self.isotonic_regressor, 'iso_regressor.joblib')

    def predict_iso(self, data_path):
        if not self.base_preds:
            self.run_base_models(data_path)
        self.meta = self.weighted_avg(data_path)
        self.isotonic_regressor = joblib.load('iso_regressor.joblib')
        iso_preds = pd.DataFrame(
            self.isotonic_regressor.predict(self.meta),
            columns=['Isotonic_Prediction']
        )
        return iso_preds


if __name__ == "__main__":

    path_to_data = './data'
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_path = os.path.join(path_to_data, train_file)
    test_path = os.path.join(path_to_data, test_file)

    print(f"Loading data :")
    # trainset = pd.read_csv(train_path)
    testset = pd.read_csv(test_path)

    # y = trainset['utility_agent1']
    # X = trainset.drop(columns=['utility_agent1'])
    y_test = testset['utility_agent1']
    X_test = testset.drop(columns=['utility_agent1'])
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # need paths not data frames .
    ensemble = Ensemble()
    ensemble_pred_avg = ensemble.weighted_avg(test_file, weights=None)
    print("Base Model Predictions (first 5):")
    print(ensemble.base_preds[0].head(10).values.reshape(-1))

    print("\nWeighted Average Predictions (first 5):", ensemble_pred_avg)
    # ensemble_avg_rsme = np.sqrt(mean_squared_error(y_test, ensemble_pred_avg))
    # print("Weighted Average RMSE:", ensemble_avg_rsme)

    # ensemble.fit_iso(train_file, y)
    # ensemble_predictions_iso = ensemble.predict_iso('demo.csv')
    # print("\nIsotonic Ensemble Predictions (first 5):", ensemble_predictions_iso[:5])
    # ensemble_rmse_iso = np.sqrt(mean_squared_error(y_test, ensemble_predictions_iso))
    # print("Isotonic Ensemble RMSE:", ensemble_rmse_iso)

