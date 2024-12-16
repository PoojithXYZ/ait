import pandas as pd
import numpy as np
import polars as pl
import os
import catboost
import sklearn
from catboost import CatBoostRegressor, cv, Pool
from sklearn.model_selection import cross_val_score, KFold
from catboost import Pool




model = None  # var to store the trained model




print("Begin Training")

path_to_data = '~/projects/ait/data/'
# Load train data
print("Loading training data...")
train_data = pl.read_csv(f'{path_to_data}train.csv')
print("Training data loaded.")

# drop cols with missing values in target col
train_data = train_data.drop_nulls(subset=['utility_agent1'])
print("Removed rows with missing values in the target column.")
train_df = train_data.to_pandas()





# selecting numerical columns
numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# fill missing values with mean in numerical columns
if numeric_cols:
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
print("Missing values filled with means in numerical columns.")

# Handle missing values in categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    mode_series = train_df[col].mode().dropna()
    if not mode_series.empty:
        mode = mode_series.iloc[0]
    else:
        mode = 'missing'
    train_df[col] = train_df[col].fillna(mode)
    train_df[col] = train_df[col].astype(str)
print("Missing values in categorical columns handled.")

# Define the target variable and columns to drop
target = 'utility_agent1'
cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', target]

# separate attributes and target
X = train_df.drop(columns=cols_to_drop, axis=1)
y = train_df[target]
print("attributes and target separated.")

# ensure categorical columns are in X
categorical_cols = [col for col in categorical_cols if col in X.columns]

# Convert categorical columns to uniform data type. here : string
for col in categorical_cols:
    X[col] = X[col].astype(str)

# Prepare data for CatBoost
categorical_features_indices = [X.columns.get_loc(col) for col in categorical_cols]
print("Categorical feature indices obtained.")





# default parameters in a dictionary
params = {
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
    'allow_writing_files':False
}
print("catboost parameters set.")

# Prepare training data
train_pool = Pool(data=X, label=y, cat_features=categorical_features_indices)
print("Training pool prepared.")

# Initialize model object and train the model
print("Training the model...")
model = CatBoostRegressor(**params)
model.fit(train_pool)
print("Training has completed.")




model.save_model('cat_OF')


model = CatBoostRegressor()
model.load_model('cat_OF')




test_df = pl.read_csv(f'{path_to_data}test.csv')
test_df = test_df.to_pandas()

categorical_cols = test_df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    test_df[col] = test_df[col].fillna('missing').astype(str)

cols_to_drop = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1']
X_test = test_df.drop(cols_to_drop, axis=1, errors='ignore')

categorical_cols = [col for col in categorical_cols if col in X_test.columns]
for col in categorical_cols:
    X_test[col] = X_test[col].astype(str)

predictions = model.predict(X_test)

table = pl.DataFrame({'Id': test_df['Id'], 'utility_agent1': 0})
result = table.with_columns(pl.Series('utility_agent1', predictions))

print(result)


