import joblib
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

XGB_MODEL_PATH = "saved_models/xgboost_model.pkl"
SELECTOR_PATH = "saved_models/selector.pkl"
FEATURE_NAMES_PATH = "saved_models/feature_names.pkl"

# ==========================================================
# Load XGBoost Model
# ==========================================================

def load_xgboost_model():

    return joblib.load(XGB_MODEL_PATH)


# ==========================================================
# Load Feature Selector
# ==========================================================

def load_selector():

    return joblib.load(SELECTOR_PATH)


# ==========================================================
# Load Selected Feature Names
# ==========================================================

def load_feature_names():

    return joblib.load(FEATURE_NAMES_PATH)


# ==========================================================
# Feature Importance
# ==========================================================

def get_feature_importance():

    model = load_xgboost_model()

    feature_names = load_feature_names()

    importance = model.feature_importances_

    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importance

    })

    importance_df = importance_df.sort_values(

        by="Importance",
        ascending=False

    ).reset_index(drop=True)

    return importance_df


# ==========================================================
# Top Biomarkers
# ==========================================================

def get_top_biomarkers(top_k=10):

    importance_df = get_feature_importance()

    return importance_df.head(top_k)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    biomarkers = get_top_biomarkers()

    print("\nTop 10 Radiomics Biomarkers\n")

    print(biomarkers)