"""
multi_model_trainer.py
----------------------
Author: Aadi Bansal
Description: 
    Generic Machine Learning training script that:
    - Loads any CSV dataset.
    - Handles missing values automatically.
    - Splits data into train/test sets.
    - Applies preprocessing (scaling + encoding).
    - Trains multiple models:
        Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, KNN.
    - Evaluates each model on Accuracy, F1, ROC-AUC.
    - Saves the best model to disk.

Usage:
    python multi_model_trainer.py path_to_dataset.csv target_column_name

Example:
    python multi_model_trainer.py data.csv Outcome
"""

# ========== IMPORTS ==========
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ========== STEP 1: LOAD DATA ==========
if len(sys.argv) != 3:
    print("Usage: python multi_model_trainer.py <dataset.csv> <target_column>")
    sys.exit(1)

file_path = sys.argv[1]
target_col = sys.argv[2]

# Read dataset
df = pd.read_csv(file_path)
print("\n✅ Data Loaded Successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ========== STEP 2: BASIC CLEANING ==========
# Drop duplicates if any
df = df.drop_duplicates()

# Drop rows where target is missing
df = df.dropna(subset=[target_col])

# Split into features & target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numerical & categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

print(f"\n🔍 Numeric Columns: {list(num_cols)}")
print(f"🔍 Categorical Columns: {list(cat_cols)}")

# ========== STEP 3: TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train size: {X_train.shape}, Test size: {X_test.shape}")

# ========== STEP 4: PREPROCESSING PIPELINE ==========
# Numeric pipeline → fill missing values + scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline → fill missing values + encode
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ========== STEP 5: DEFINE MODELS ==========
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42)
}

# ========== STEP 6: TRAIN & EVALUATE EACH MODEL ==========
results = []

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")

    # Build full pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Fit model
    pipe.fit(X_train, y_train)

    # Predict on test set
    preds = pipe.predict(X_test)

    # Try to get probabilities (for ROC-AUC)
    try:
        probs = pipe.predict_proba(X_test)[:, 1]
    except:
        probs = None

    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    auc = roc_auc_score(y_test, probs) if probs is not None else np.nan

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    print(f"✅ {name} Results:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}" if probs is not None else "   ROC-AUC: N/A")

# ========== STEP 7: COMPARE RESULTS ==========
result_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n📈 Model Performance Comparison:")
print(result_df)

# Save results to CSV
result_df.to_csv("model_results.csv", index=False)
print("\n💾 Saved model performance to 'model_results.csv'")

# ========== STEP 8: SAVE BEST MODEL ==========
best_model_name = result_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\n🏆 Best model: {best_model_name}")

# Re-train best model on full data
best_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])
best_pipe.fit(X, y)

joblib.dump(best_pipe, "best_model.joblib")
print("💾 Best model saved as 'best_model.joblib'")

# ========== STEP 9: HOW TO USE SAVED MODEL ==========
"""
Example usage in your code:

import joblib
import pandas as pd

# Load saved model
model = joblib.load('best_model.joblib')

# Prepare new data (must have same columns as training data)
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    ...
})

# Predict
pred = model.predict(new_data)
print("Prediction:", pred)
"""

print("\n🎯 DONE! All models trained and best one saved successfully.")
