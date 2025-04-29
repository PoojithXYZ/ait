import pandas as pd
import numpy as np
import polars as pl
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import root_mean_squared_error as rmse


class My_XGB_Model:
    def __init__(self, path_to_data='~/projects/ait/data/'):
        self.path_to_data = path_to_data
        self.model = None
        self.categorical_cols = None
        self.label_encoders = {}
        self.train_columns = None  # Store training columns

        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 70,
            'learning_rate': 0.08780929167510496,
            'max_depth': 7,
            'reg_alpha': 0.0,
            'reg_lambda': 0.04063851295419513,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 1,
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 1
        }

    def train(self):
        print("Loading the data...")
        train_df = pl.read_csv(os.path.join(self.path_to_data, 'train.csv')).to_pandas()
        train_df = train_df.dropna(subset=['utility_agent1'])
        print('Dropped null values.')
        numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
        print('seperating columns into numeric and categorical...')
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            mode_series = train_df[col].mode().dropna()
            if not mode_series.empty:
                mode = mode_series.iloc[0]
            else:
                mode = 'missing'
            train_df[col] = train_df[col].fillna(mode)
            train_df[col] = train_df[col].astype(str)

        self.categorical_cols = [col for col in categorical_cols if col in train_df.columns]

        cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', 'utility_agent1', 'Id'] # Added 'Id' for consistency
        X = train_df.drop(columns=cols_to_drop, axis=1, errors='ignore')
        y = train_df['utility_agent1']

        # Encode categorical features
        X_encoded = X.copy()
        for col in self.categorical_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self.label_encoders[col] = le

        self.train_columns = X_encoded.columns.tolist() # Store the columns after encoding
        X_train, X_eval, y_train, y_eval = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        print("Training the model...")
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train,  
            y_train,  
            eval_set=[(X_eval, y_eval)],  
            verbose=100
        )
        print("Training completed.")

        self.model.save_model('xgb_OF.json')

    def predict(self, test_data=None):
        self.model = xgb.XGBRegressor()
        self.model.load_model('xgb_OF.json')

        if test_data is None:
            test_df = pl.read_csv(os.path.join(self.path_to_data, 'sample.csv')).to_pandas()
        else:
            test_df = pl.read_csv(os.path.join(self.path_to_data, test_data)).to_pandas()

        numeric_cols = test_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())

        self.categorical_cols = test_df.select_dtypes(include=['object']).columns.tolist()
        for col in self.categorical_cols:
            mode_series = test_df[col].mode().dropna()
            if not mode_series.empty:
                mode = mode_series.iloc[0]
            else:
                mode = 'missing'
            test_df[col] = test_df[col].fillna(mode)
            test_df[col] = test_df[col].astype(str)

        cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', 'Id'] # Added 'Id' for consistency
        X_test = test_df.drop(columns=cols_to_drop, axis=1, errors='ignore')

        # Encode categorical features in the test set
        X_test_encoded = X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_encoded.columns and col in self.label_encoders:
                X_test_encoded[col] = self.label_encoders[col].transform(X_test_encoded[col])
            elif col in X_test_encoded.columns and col not in self.label_encoders:
                # Handle cases where a new category appears in test data
                X_test_encoded[col] = X_test_encoded[col].astype('category').cat.codes
            elif col not in X_test_encoded.columns and col in self.label_encoders:
                # Handle cases where a categorical column from training is missing in test
                X_test_encoded[col] = np.nan # Or some other appropriate handling
                X_test_encoded[col] = X_test_encoded[col].fillna(-1) # Example filling

        # Ensure consistent column order
        if self.train_columns is not None:
            missing_cols_train = set(self.train_columns) - set(X_test_encoded.columns)
            for c in missing_cols_train:
                X_test_encoded[c] = 0 # Fill with a default value

            missing_cols_test = set(X_test_encoded.columns) - set(self.train_columns)
            if missing_cols_test:
                print(f"Warning: Extra columns in test data: {missing_cols_test}")

            # Reorder columns to match training data
            X_test_encoded = X_test_encoded[self.train_columns]

        predictions = self.model.predict(X_test_encoded)
        result = pd.DataFrame({
            # 'Id': test_df['Id'],
            'utility_agent1': predictions
        })

        return result

if __name__ == "__main__":
    model = My_XGB_Model()
    # model.train()
    result = model.predict()
    print(result)
    # result = model.predict(test_data='test.csv')
    # print(result)

