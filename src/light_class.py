import os
import polars as pl
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm
from lightgbm import LGBMRegressor
import json
import pickle


class My_Light_Model:
    def __init__(self):
        self.path_to_data = '~/projects/ait/data/'
        self.target = 'utility_agent1'
        self.train_cols = []
        self.categorical_features = []
        self.Light_Models = []
        self.counter = 0
        self.rmses = []

    def train(self):
        print("Starting to train...")

        # Loading data
        train = pd.read_csv(f'{self.path_to_data}train.csv')
        y_train = train[self.target]

        print('dropping columns...')
        # Drop columns with unique values < 2 and unwanted columns
        cols_to_drop = [col for col in train.columns if train[col].nunique() < 2]
        cols_to_drop.extend(['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', self.target])
        train = train.drop(columns=cols_to_drop)

        print('selecting categorical columns...')
        # select categorical columns
        cols_with_object_as_val = train.select_dtypes(include='object').columns.tolist()
        categorical_cols = [col for col in train.columns if train[col].nunique() == 2 and train[col].dtype in ['object', 'int64', 'float64']]

        # Convert selected columns to category type
        train[cols_with_object_as_val + categorical_cols] = train[cols_with_object_as_val + categorical_cols].astype('category')
        self.train_cols = train.columns
        self.categorical_features = cols_with_object_as_val + categorical_cols

        print('Preprocessing Done.')
        # Reset models list
        self.Light_Models = []
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.rmses = []

        print('Begin training...')
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
            X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMRegressor(random_state=42, verbose=1)
            model.fit(X_train, y_train_fold, categorical_feature=self.categorical_features)
            # model.predict(X_val)
            self.Light_Models.append(model)
            model.booster_.save_model(f"light_OF_k{fold+1}.txt")

            val_preds = model.predict(X_val)
            rmse = mean_squared_error(y_val_fold, val_preds, squared=False)
            print(f"Fold {fold + 1} : {rmse}")
            self.rmses.append(rmse)

        print("Mean RMSE:", np.mean(self.rmses))
        print("Training completed.")

    def predict(self, test_data='test.csv'):
        test = pd.read_csv(f'{self.path_to_data}{test_data}')
        y_train = test[self.target]

        print('dropping columns...')
        # Drop columns with unique values < 2 and unwanted columns
        cols_to_drop = [col for col in test.columns if test[col].nunique() < 2]
        cols_to_drop.extend(['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', self.target])
        test = test.drop(columns=cols_to_drop)

        print('selecting categorical columns...')
        # select categorical columns
        cols_with_object_as_val = test.select_dtypes(include='object').columns.tolist()
        categorical_cols = [col for col in test.columns if test[col].nunique() == 2 and test[col].dtype in ['object', 'int64', 'float64']]

        # Convert selected columns to category type
        test[cols_with_object_as_val + categorical_cols] = test[cols_with_object_as_val + categorical_cols].astype('category')
        self.test_cols = test.columns
        self.categorical_features = cols_with_object_as_val + categorical_cols


        # # categorical_cols = [col for col in test.columns if test[col].nunique() == 2 and test[col].dtype in ['object', 'int64', 'float64']]
        # test[self.categorical_features] = test[self.categorical_features].astype('category')
        # missing_cols = [col for col in self.train_cols if col not in test.columns]
        # for col in missing_cols:
        #     test[col] = 0
        # test = test[self.train_cols]

        if not self.Light_Models:
            for i in range(5):
                model = lightgbm.Booster(model_file=f"light_OF_k{i+1}.txt")
                self.Light_Models.append(model)

        def mean_of_models(data, models):
            return np.mean([model.predict(data) for model in models], axis=0)

        predictions = mean_of_models(test, self.Light_Models)
        result = pd.DataFrame({
            # 'Id': test['Id'], 
            'utility_agent1': predictions
        }, index=test.index)
        return result


if __name__ == "__main__":
    model = My_Light_Model()
    # print(model.train()
    print(model.predict())
    # print(model.predict('test.csv'))


