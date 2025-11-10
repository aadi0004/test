"""
ml_end_to_end_pipeline.py
-------------------------
🔥 End-to-End Machine Learning Pipeline in a Single File

Features:
  ✅ Data loading & cleaning
  ✅ Feature engineering (encoding, scaling, imputation)
  ✅ Train/test split
  ✅ Train multiple models (Logistic Regression, Random Forest, XGBoost, SVM)
  ✅ Evaluate with accuracy, F1-score, ROC-AUC
  ✅ Compare models automatically
  ✅ Feature importance (where supported)
  ✅ Explainability using SHAP
  ✅ Save best model to disk
  ✅ Load model and make predictions

Usage:
  python ml_end_to_end_pipeline.py data.csv target_column
"""

# =============================
# 📦 Import Dependencies
# =============================
import sys
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# =============================
# ⚙️ Load and Prepare Data
# =============================
def load_data(file_path, target_col):
    """
    Loads dataset, separates features and target, and handles missing data.
    """
    df = pd.read_csv(file_path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: {target_col}")
    
    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify column types
    num_cols = X.select_dtypes(include=['int64','float64']).columns
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns
    print(f"🔢 Numeric cols: {list(num_cols)}")
    print(f"🔤 Categorical cols: {list(cat_cols)}")

    return X, y, num_cols, cat_cols

# =============================
# 🧠 Preprocessing Pipeline
# =============================
def build_preprocessor(num_cols, cat_cols):
    """
    Returns a ColumnTransformer for numeric and categorical preprocessing.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor

# =============================
# 🤖 Train Multiple Models
# =============================
def train_models(preprocessor, X_train, y_train):
    """
    Trains multiple ML algorithms and returns fitted pipelines.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42)
    }

    pipelines = {}
    for name, model in models.items():
        print(f"🚀 Training {name}...")
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines

# =============================
# 📈 Evaluate Models
# =============================
def evaluate_models(pipelines, X_test, y_test):
    """
    Evaluates each trained model and returns a comparison table.
    """
    results = []
    for name, pipe in pipelines.items():
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps["model"], "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        auc = roc_auc_score(y_test, probs) if probs is not None else np.nan

        print(f"✅ {name} | Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
        results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": auc})

    result_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n📊 Model Comparison:\n", result_df)
    return result_df

# =============================
# 🌟 Save Best Model
# =============================
def save_best_model(result_df, pipelines):
    """
    Saves the best-performing model to disk.
    """
    best_model_name = result_df.iloc[0]["Model"]
    best_pipe = pipelines[best_model_name]
    joblib.dump(best_pipe, "best_model.joblib")
    print(f"💾 Best model ({best_model_name}) saved to best_model.joblib")
    return best_pipe, best_model_name

# =============================
# 🔍 Explain Model with SHAP
# =============================
def explain_model(pipe, X_sample):
    """
    Generates SHAP summary plot for tree-based models.
    """
    model = pipe.named_steps["model"]
    print("\n🔎 Generating SHAP explanations...")
    try:
        explainer = shap.TreeExplainer(model)
        X_transformed = pipe.named_steps["pre"].transform(X_sample)
        shap_values = explainer.shap_values(X_transformed)
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP Values)")
        plt.show()
    except Exception as e:
        print("⚠️ SHAP explanation skipped:", e)

# =============================
# 🧾 Make Predictions
# =============================
def predict_new(sample_dict):
    """
    Load saved model and make predictions on new sample data.
    """
    model = joblib.load("best_model.joblib")
    df = pd.DataFrame([sample_dict])
    pred = model.predict(df)[0]
    print(f"🎯 Prediction: {pred}")
    return pred

# =============================
# 🏁 Main Execution
# =============================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ml_end_to_end_pipeline.py <data.csv> <target_column>")
        sys.exit()

    file_path, target_col = sys.argv[1], sys.argv[2]

    # Step 1: Load data
    X, y, num_cols, cat_cols = load_data(file_path, target_col)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"📂 Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Step 3: Build preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Step 4: Train models
    pipelines = train_models(preprocessor, X_train, y_train)

    # Step 5: Evaluate all
    result_df = evaluate_models(pipelines, X_test, y_test)

    # Step 6: Save best model
    best_pipe, best_name = save_best_model(result_df, pipelines)

    # Step 7: Explain best model
    explain_model(best_pipe, X_test.sample(min(100, len(X_test)), random_state=42))

    # Step 8: Predict on a sample (for demo)
    print("\n🧩 Demo prediction:")
    sample = X_test.sample(1, random_state=42).to_dict(orient="records")[0]
    predict_new(sample)



# 📦 requirements.txt
# pandas
# numpy
# scikit-learn
# matplotlib
# xgboost
# shap
# joblib

# ▶️ How to Run
# python ml_end_to_end_pipeline.py data.csv target_column



# ✅ Loaded data: 1000 rows, 10 columns
# 🔢 Numeric cols: ['age','income','balance']
# 🔤 Categorical cols: ['gender','city']

# 🚀 Training Logistic Regression...
# 🚀 Training Random Forest...
# ...

# ✅ Random Forest | Acc=0.92 | F1=0.91 | AUC=0.95
# 📊 Model Comparison:
#              Model  Accuracy    F1  ROC-AUC
# 1   Random Forest      0.92  0.91     0.95
# 3          XGBoost      0.91  0.90     0.94
# ...

# 💾 Best model (Random Forest) saved to best_model.joblib
# 🔎 Generating SHAP explanations...
# 🎯 Prediction: 1
