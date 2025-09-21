import pandas as pd
import numpy as np
import polars as pl
import os
from catboost import CatBoostRegressor, Pool


class My_Cat_B_Model:
    def __init__(self, path_to_data='~/projects/ait/data/'):
        self.path_to_data = path_to_data
        self.model = None
        self.categorical_cols = None
        
        self.params = {
            'iterations': 700,
            'learning_rate': 0.08780929167510496,
            'depth': 7,
            'l2_leaf_reg': 0.04063851295419513,
            'bagging_temperature': 0.8072638982547167,
            'random_strength': 0.7584336734894706,
            'border_count': 33,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'task_type': 'CPU',
            'verbose': 100,
            'allow_writing_files': False
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

        cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', 'utility_agent1']
        X = train_df.drop(columns=cols_to_drop, axis=1)
        y = train_df['utility_agent1']

        categorical_features_indices = [X.columns.get_loc(col) for col in self.categorical_cols if col in X.columns]

        train_pool = Pool(data=X, label=y, cat_features=categorical_features_indices)

        print("Training the model...")
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_pool)
        print("Training completed.")

        self.model.save_model('cat_OF')

    def predict(self, test_data='test.csv'):
        self.model = CatBoostRegressor()
        self.model.load_model('cat_OF')
        print(test_data, '------------------')
        if test_data is None:
            test_df = pl.read_csv(os.path.join(self.path_to_data, 'sample.csv')).to_pandas()
        else:
            test_df = pl.read_csv(test_data).to_pandas()

        numeric_cols = test_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())

        categorical_cols = test_df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            mode_series = test_df[col].mode().dropna()
            if not mode_series.empty:
                mode = mode_series.iloc[0]
            else:
                mode = 'missing'
            test_df[col] = test_df[col].fillna(mode)
            test_df[col] = test_df[col].astype(str)

        cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1']
        X_test = test_df.drop(columns=cols_to_drop, axis=1, errors='ignore')

        predictions = self.model.predict(X_test)
        result = pd.DataFrame({
            # 'Id': test_df['Id'],
            'utility_agent1': predictions
        })
        # print(result.shape)
        return result


if __name__ == "__main__":
    model = My_Cat_B_Model()
    model.train()    
    result = model.predict('test.csv')
    print(result)
    # result = model.predict(test_data='test.csv')
    # print(result)

