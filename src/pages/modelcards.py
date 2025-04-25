import streamlit as st
import os


MODELS = [
    {
        "name": "CatBoost",
        "image_path": 'src/pages/catboost.png',
        "hyperparameters": {
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
    },
    {
        "name": "XGBoost",
        "image_path": 'src/pages/xgboost.png',
        "hyperparameters": {
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
    },
    {
        "name": "LightGBM",
        "image_path": 'src/pages/lightgbm.png',
        "hyperparameters": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
        }
    }
]


def multiple_model_cards_page(models_data):
    st.set_page_config(page_title="Model Catalog", layout="wide")
    st.title("Model Catalog")

    for model in models_data:
        st.markdown("---")

        col1, col2 = st.columns([3, 4])

        with col1:
            st.header(f"Model : {model['name']}")

        with col2:
            st.image(model["image_path"], width=300)


        st.write("**Hyperparameters :**")
        if model.get("hyperparameters"):
             for param, value in model["hyperparameters"].items():
                 st.write(f"- **{param} :** {value}")
        else:
             st.write("No hyperparameters specified.")

        st.markdown("---")


if __name__ == "__main__":
    multiple_model_cards_page(MODELS)
