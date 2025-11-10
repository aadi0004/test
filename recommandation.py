"""
advanced_ml_systems.py
----------------------
🔥 All-in-One Advanced ML File
Includes:
    1️⃣ Time Series Forecasting
    2️⃣ Recommender System
    3️⃣ Anomaly Detection

Each section is self-contained and can be run independently.
Just scroll to the bottom and uncomment which one you want to execute.
"""

# ================================================================
# 📈 1️⃣ TIME SERIES FORECASTING SECTION
# ================================================================

"""
Goal:
Predict future sales/temperature/stock data using Prophet or ARIMA.

Dataset: Must have columns ['ds', 'y']
    ds = date column
    y  = target numeric value
"""

# --- Import libraries ---
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# Try to import Prophet; if not available, fallback to ARIMA
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    from statsmodels.tsa.arima.model import ARIMA
    PROPHET_AVAILABLE = False

def run_time_series_forecasting(csv_path="timeseries_data.csv"):
    df = pd.read_csv(csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    print(f"✅ Loaded time series data: {df.shape[0]} records")

    if PROPHET_AVAILABLE:
        print("⏳ Training Prophet model...")
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot results
        model.plot(forecast)
        plt.title("📈 Prophet Forecast (Next 30 Days)")
        plt.show()

        forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv("forecast_output.csv", index=False)
        print("💾 Forecast saved to forecast_output.csv")

    else:
        print("⚠️ Prophet not found, using ARIMA instead...")
        df = df.set_index('ds')
        model = ARIMA(df['y'], order=(2,1,2))
        res = model.fit()
        pred = res.get_forecast(steps=30)
        forecast = pred.summary_frame()
        print(forecast.head())
        forecast[['mean']].to_csv("forecast_output.csv")
        print("💾 Forecast saved using ARIMA.")


# ================================================================
# 💡 2️⃣ RECOMMENDER SYSTEM SECTION
# ================================================================

"""
Goal:
Recommend items to users based on their historical ratings.

Dataset: Must have columns ['user_id', 'item_id', 'rating']
"""

from sklearn.metrics.pairwise import cosine_similarity

def run_recommender_system(csv_path="ratings_data.csv", top_n=5):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded rating data: {df.shape}")

    # Create user-item matrix
    pivot = df.pivot_table(index='user_id', columns='item_id', values='rating')
    pivot.fillna(0, inplace=True)

    # Compute similarity between items
    item_sim = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)

    def recommend_items(user_id):
        """Generate item recommendations for a given user."""
        user_ratings = pivot.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        scores = sim_df[rated_items].mean(axis=1).sort_values(ascending=False)
        recs = scores.drop(rated_items).head(top_n)
        return recs.index.tolist()

    # Example: Recommend for one user
    sample_user = pivot.index[0]
    recs = recommend_items(sample_user)
    print(f"🎯 Recommendations for User {sample_user}: {recs}")

    # Save similarity matrix
    sim_df.to_csv("item_similarity_matrix.csv")
    print("💾 Item similarity matrix saved.")

# ================================================================
# 🚨 3️⃣ ANOMALY DETECTION SECTION
# ================================================================

"""
Goal:
Detect anomalies/outliers in numeric data using IsolationForest and Z-score.

Dataset: Must have a numeric column to detect anomalies (e.g., 'value')
"""

from sklearn.ensemble import IsolationForest

def run_anomaly_detection(csv_path="anomaly_data.csv", column="value"):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset: {df.shape}")

    # Basic check
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    # Using IsolationForest
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[[column]])

    # Using simple Z-score method
    df['zscore'] = (df[column] - df[column].mean()) / df[column].std()
    df['z_anomaly'] = np.where(abs(df['zscore']) > 3, -1, 1)

    anomalies = df[df['anomaly'] == -1]
    print(f"🚨 Detected {len(anomalies)} anomalies via IsolationForest.")
    df.to_csv("anomaly_results.csv", index=False)
    print("💾 Saved labeled dataset to anomaly_results.csv")

    # Visualization
    plt.figure(figsize=(10,5))
    plt.scatter(df.index, df[column], c=(df['anomaly']==-1), cmap='coolwarm')
    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.show()


# ================================================================
# 🏁 MAIN EXECUTION CONTROL
# ================================================================
if __name__ == "__main__":
    """
    Uncomment one of the following lines to run a specific module.
    Each expects a CSV with specific columns.
    """

    # -------- TIME SERIES --------
    # run_time_series_forecasting("timeseries_data.csv")

    # -------- RECOMMENDER --------
    # run_recommender_system("ratings_data.csv", top_n=5)

    # -------- ANOMALY DETECTION --------
    # run_anomaly_detection("anomaly_data.csv", column="value")



# pandas
# numpy
# matplotlib
# scikit-learn
# prophet
# statsmodels




# recommandation with fast api

"""
recommender_api.py
------------------
🎯 End-to-End Recommender System + API Deployment

✅ Loads dataset (user-item ratings or item metadata)
✅ Trains recommender using item similarity
✅ Saves the trained model
✅ Exposes FastAPI endpoints:
      - /train       → Train model
      - /recommend   → Get recommendations for an item

Example dataset (movies.csv):
------------------------------------------------
item_id,title,genre,description
1,Inception,Sci-Fi,thrilling mind-bending dream heist
2,Interstellar,Sci-Fi,space travel and time relativity
3,The Dark Knight,Action,batman fights crime in gotham
4,The Prestige,Drama,two magicians rivalry and illusion
5,Avatar,Sci-Fi,humans explore alien world Pandora
------------------------------------------------
"""

# =============================
# 📦 Import Libraries
# =============================
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import os

# =============================
# ⚙️ Global Variables
# =============================
MODEL_FILE = "recommender_model.joblib"
DATA_FILE = "items.csv"

# =============================
# 🧠 TRAINING FUNCTION
# =============================
def train_recommender(data_path=DATA_FILE):
    """
    1️⃣ Load items dataset
    2️⃣ Create text embeddings using TF-IDF
    3️⃣ Compute cosine similarity
    4️⃣ Save model components
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {df.shape[0]} items.")

    # Combine relevant text features (title + description + genre)
    df["text"] = (
        df["title"].astype(str) + " " +
        df["genre"].astype(str) + " " +
        df["description"].astype(str)
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    # Compute cosine similarity between all items
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Save components
    model_data = {
        "items": df,
        "vectorizer": vectorizer,
        "similarity": similarity_matrix
    }
    joblib.dump(model_data, MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")

# =============================
# 🎯 RECOMMEND FUNCTION
# =============================
def get_recommendations(item_query: str, top_n: int = 5):
    """
    1️⃣ Load trained model
    2️⃣ Match query item or find closest by text similarity
    3️⃣ Return top N similar items
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Model not trained yet. Run /train first.")

    model_data = joblib.load(MODEL_FILE)
    df = model_data["items"]
    vectorizer = model_data["vectorizer"]
    similarity = model_data["similarity"]

    # Try to find the item directly
    matches = df[df["title"].str.lower().str.contains(item_query.lower())]
    if matches.empty:
        # If not found, treat query as free text and find best match
        query_vec = vectorizer.transform([item_query])
        all_text_vec = vectorizer.transform(df["text"])
        scores = cosine_similarity(query_vec, all_text_vec).flatten()
        idx = np.argmax(scores)
        print(f"⚠️ Item not found, using closest match: {df.iloc[idx]['title']}")
    else:
        idx = matches.index[0]

    # Get top-N similar items
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_indices = [i for i, _ in sim_scores]
    recs = df.iloc[rec_indices][["title", "genre", "description"]]

    return recs.to_dict(orient="records")

# =============================
# 🌐 FASTAPI APP
# =============================
app = FastAPI(
    title="Recommender System API",
    description="Train and query a simple content-based recommender.",
    version="1.0"
)

class TrainRequest(BaseModel):
    data_path: str = DATA_FILE

class RecommendRequest(BaseModel):
    query: str
    top_n: int = 5

@app.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        train_recommender(req.data_path)
        return {"message": f"Model trained successfully from {req.data_path}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    try:
        recs = get_recommendations(req.query, req.top_n)
        return {"query": req.query, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Recommender System API is running."}

# =============================
# ▶️ MAIN ENTRY POINT
# =============================
if __name__ == "__main__":
    # You can run this directly: python recommender_api.py
    # Ensure items.csv exists before running /train endpoint.
    uvicorn.run(app, host="0.0.0.0", port=8000)




# .yml

# name: 🚀 CI/CD Pipeline - Recommender System

# on:
#   push:
#     branches:
#       - main
#   pull_request:
#     branches:
#       - main

# jobs:
#   build-test-deploy:
#     runs-on: ubuntu-latest

#     env:
#       PYTHON_VERSION: 3.10
#       APP_PORT: 8000
#       SSH_HOST: ${{ secrets.SSH_HOST }}
#       SSH_USER: ${{ secrets.SSH_USER }}
#       SSH_KEY: ${{ secrets.SSH_KEY }}

#     steps:
#       # Step 1: Checkout repository
#       - name: 🧭 Checkout code
#         uses: actions/checkout@v4

#       # Step 2: Setup Python
#       - name: 🐍 Setup Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: ${{ env.PYTHON_VERSION }}

#       # Step 3: Install dependencies
#       - name: 📦 Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install -r requirements.txt

#       # Step 4: Linting + basic health check
#       - name: 🧪 Lint and API health test
#         run: |
#           pip install flake8 requests
#           echo "Running flake8 lint check..."
#           flake8 recommender_api.py || true
#           echo "✅ Basic syntax OK."

#       # Step 5: Docker build
#       - name: 🐳 Build Docker image
#         run: |
#           docker build -t recommender-api:latest .

#       # Step 6: Deploy via SSH to your server
#       - name: 🚀 Deploy to Remote Server
#         if: github.ref == 'refs/heads/main'
#         run: |
#           echo "${{ env.SSH_KEY }}" > private_key.pem
#           chmod 600 private_key.pem

#           echo "Connecting to ${{ env.SSH_HOST }} ..."
#           ssh -o StrictHostKeyChecking=no -i private_key.pem ${{ env.SSH_USER }}@${{ env.SSH_HOST }} << 'EOF'
#             set -e
#             echo "Pulling latest changes..."
#             cd ~/recommender_project || mkdir ~/recommender_project && cd ~/recommender_project
#             git pull origin main || true
#             echo "Stopping old container..."
#             docker stop recommender-api || true
#             docker rm recommender-api || true
#             echo "Building new container..."
#             docker build -t recommender-api .
#             echo "Starting new container..."
#             docker run -d -p 80:8000 --name recommender-api recommender-api
#           EOF
