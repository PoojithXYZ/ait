import streamlit as st
import pandas as pd

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cat_class import My_Cat_B_Model
from xgboost_class import My_XGB_Model
from light_class import My_Light_Model
from ensemble import Ensemble

path_to_data = '/home/poojith-xyz/projects/ait/data/'

st.title("Compare Your MCTS Agents")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:

    with open(os.path.join(path_to_data, 'demo.csv'), "w", encoding="utf-8") as f:
        content = uploaded_file.getvalue()
        # Decode the bytes to string and write to file
        content = content.decode("utf-8")
        f.write(content)
    print("Uploaded file:", uploaded_file)

    df = pd.read_csv(uploaded_file)
    st.write("DataFrame:")
    st.write(df.head())

    ensemble_model = Ensemble()
    ensemble_model.run_base_models('demo.csv')
    avgs = ensemble_model.weighted_avg('demo.csv')
    iso_preds = ensemble_model.predict_iso('demo.csv')

    st.write("Generating predictions...")
    st.success("Predictions generated.") # Simple success message

    st.write("Combining predictions...")
    predictions_df = pd.DataFrame({
        'CatBoost_Predictions': ensemble_model.base_preds[0].values.reshape(-1),
        'XGBoost_Predictions': ensemble_model.base_preds[1].values.reshape(-1),
        'LightGBM_Predictions': ensemble_model.base_preds[2].values.reshape(-1),
        'Average_Predictions': avgs.values.reshape(-1),
        'Isotonic_Regression_Predictions': iso_preds.values.reshape(-1)
    })
    print(predictions_df)

    # --- Display Combined Predictions ---
    st.write("Model Predictions:")
    st.dataframe(predictions_df) # Display the predictions table
else:
    st.write("Please upload a CSV file to see the data.")
