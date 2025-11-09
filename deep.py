#!/usr/bin/env python3
"""
deep_learning_prod_all_in_one.py

All-in-one production-ready deep learning & ML ops template.

Sections:
- Header & requirements
- Utilities & config
- ETL / Data helpers
- Image classifier (ResNet transfer learning, PyTorch)
- Text classifier (BERT fine-tune via Hugging Face Trainer)
- Time-series forecaster (LSTM with PyTorch)
- Autoencoder anomaly detector (tabular, PyTorch)
- MLflow integration helpers
- Drift detection & monitoring utilities
- FastAPI server to serve models & management endpoints

Usage:
    # Train image classifier
    python deep_learning_prod_all_in_one.py train_image --data_dir data/images --epochs 5

    # Fine-tune text classifier
    python deep_learning_prod_all_in_one.py train_text --dataset data/text.csv --text_col text --label_col label

    # Run FastAPI server
    python deep_learning_prod_all_in_one.py serve

Requirements (paste into requirements.txt):
    torch>=1.13.0
    torchvision
    transformers
    datasets
    scikit-learn
    pandas
    numpy
    mlflow
    fastapi
    uvicorn
    pillow
    tqdm
    matplotlib
    seaborn
    scipy
    joblib
"""

# =========================
# HEADER: Requirements text
# =========================
REQUIREMENTS = """
torch>=1.13.0
torchvision
transformers
datasets
scikit-learn
pandas
numpy
mlflow
fastapi
uvicorn
pillow
tqdm
matplotlib
seaborn
scipy
joblib
"""

# =========================
# IMPORTS & GLOBAL CONFIG
# =========================
import os
import json
import time
import random
import argparse
import logging
from typing import List, Dict, Any, Optional

# Core data libs
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Hugging Face
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import Dataset as HFDataset, load_metric

# MLflow
import mlflow
import mlflow.pytorch

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Monitoring & stats
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Serialization
import joblib

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deep_all_in_one")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ======== Utility functions ========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ========= MLflow init =========
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)  # set if remote
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "deep_all_in_one")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# =========================
# ETL / Data helpers
# =========================
def load_tabular_csv(path: str, target_col: Optional[str] = None) -> pd.DataFrame:
    """Load tabular CSV and basic cleaning."""
    df = pd.read_csv(path)
    logger.info(f"Loaded CSV {path} shape={df.shape}")
    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    # Remove columns with all nulls
    df = df.loc[:, df.isnull().sum() < len(df)]
    if target_col and target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in {path}")
    return df

# Minimal data validator
def validate_columns(df: pd.DataFrame, required: List[str]):
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

# =========================
# IMAGE CLASSIFIER (TRANSFER LEARNING)
# =========================

# PyTorch Dataset for image folder where structure is a flat CSV mapping item -> path -> label (recommended)
class ImageCSVLoader(Dataset):
    """
    Loads images from CSV file with columns: image_path, label
    image_path can be absolute or relative to data_dir.
    """
    def __init__(self, df: pd.DataFrame, data_dir: Optional[str] = None, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir or "."
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_path"]) if not os.path.isabs(row["image_path"]) else row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label

def train_image_classifier(data_csv: str,
                           output_dir: str = "artifacts/image_model",
                           model_name: str = "resnet18",
                           epochs: int = 5,
                           batch_size: int = 32,
                           lr: float = 1e-3,
                           num_workers: int = 4):
    """
    Train a transfer-learning image classifier using torchvision models.
    data_csv: csv with columns ['image_path', 'label']
    Saves model state_dict and metadata for inference.
    """
    ensure_dir(output_dir)
    df = pd.read_csv(data_csv)
    validate_columns(df, ["image_path", "label"])

    # Prepare transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    train_ds = ImageCSVLoader(train_df, transform=train_transform)
    val_ds = ImageCSVLoader(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, int(df["label"].nunique()))
    else:
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, int(df["label"].nunique()))

    model = model.to(DEVICE)

    # Freeze base layers optionally
    for param in model.parameters():
        param.requires_grad = True  # set False to freeze, True to fine-tune

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    best_epoch = -1
    mlflow_run = mlflow.start_run(run_name=f"image_train_{int(time.time())}")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_total, v_correct = 0, 0
        v_running_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                v_loss = criterion(outputs, labels)
                v_running_loss += v_loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_loss = v_running_loss / v_total
        val_acc = v_correct / v_total

        logger.info(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save best state
            torch.save(model.state_dict(), os.path.join(output_dir, "best_image_model.pth"))
            # Save metadata
            joblib.dump({"model_name": model_name, "classes": sorted(df["label"].unique())}, os.path.join(output_dir, "meta.joblib"))
            # Log model as artifact to MLflow
            mlflow.pytorch.log_model(model, artifact_path="image_model")

        scheduler.step()

    mlflow.log_metric("best_val_acc", best_val_acc)
    mlflow.end_run()
    logger.info(f"Training complete. Best val acc: {best_val_acc} at epoch {best_epoch}")
    return os.path.join(output_dir, "best_image_model.pth")

def predict_image(model_pth: str, meta_path: str, image_path: str):
    """
    Quick inference util for image model
    """
    meta = joblib.load(meta_path)
    model_name = meta.get("model_name", "resnet18")
    classes = meta.get("classes")
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(classes))
    else:
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(torch.load(model_pth, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
        return classes[pred] if classes else int(pred)

# =========================
# TEXT CLASSIFIER (BERT via Hugging Face Trainer)
# =========================
def train_text_classifier(csv_path: str,
                          text_col: str,
                          label_col: str,
                          model_checkpoint: str = "distilbert-base-uncased",
                          output_dir: str = "artifacts/text_model",
                          epochs: int = 3,
                          batch_size: int = 16,
                          lr: float = 2e-5):
    """
    Fine-tune a Hugging Face transformer for text classification using Trainer API.
    csv_path must contain [text_col, label_col].
    """
    ensure_dir(output_dir)
    df = load_tabular_csv(csv_path)
    validate_columns(df, [text_col, label_col])
    labels = sorted(df[label_col].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df["label_id"] = df[label_col].map(label2id)

    # Convert to HF dataset
    hf_ds = HFDataset.from_pandas(df[[text_col, "label_id"]])
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_fn(ex):
        return tokenizer(ex[text_col], truncation=True, padding="max_length", max_length=256)
    hf_ds = hf_ds.map(tokenize_fn, batched=False)
    hf_ds = hf_ds.rename_column("label_id", "labels")
    hf_ds.set_format(type="torch", columns=['input_ids','attention_mask','labels'])

    # Train-test split
    split = hf_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split['train']
    val_ds = split['test']

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(labels))
    training_args = TrainingArguments(output_dir=output_dir,
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      learning_rate=lr,
                                      logging_strategy="steps",
                                      logging_steps=50,
                                      load_best_model_at_end=True)

    # Define compute_metrics
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels_arr)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels_arr, average="weighted")["f1"]
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_ds,
                      eval_dataset=val_ds,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    mlflow.start_run(run_name=f"text_train_{int(time.time())}")
    mlflow.log_param("model_checkpoint", model_checkpoint)
    mlflow.log_param("epochs", epochs)
    trainer.train()
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Log to mlflow (Hugging Face model dir zipped)
    mlflow.end_run()
    logger.info(f"Text model saved to {output_dir}")
    return output_dir, label2id

def predict_text(model_dir: str, label2id: dict, text: str, model_checkpoint: str = None):
    """
    Load text model and tokenizer from model_dir and predict a label for a given text string.
    label2id: mapping label->id, we invert it for id->label.
    """
    id2label = {v:k for k,v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return id2label[pred]

# =========================
# TIME SERIES FORECASTER (LSTM)
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int = 24):
        self.series = series.astype(np.float32)
        self.window = window

    def __len__(self):
        return max(0, len(self.series) - self.window)

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.window]
        y = self.series[idx + self.window]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # take last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

def train_lstm_forecaster(csv_path: str, value_col: str = "y", window: int = 24,
                          epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                          output_dir: str = "artifacts/ts_model"):
    """
    Train an LSTM forecaster on a single numeric column 'value_col' in csv_path.
    """
    ensure_dir(output_dir)
    df = load_tabular_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"{value_col} not in dataset")
    series = df[value_col].fillna(method="ffill").values
    # Normalize
    mean = series.mean()
    std = series.std()
    series_norm = (series - mean) / (std + 1e-8)

    ds = TimeSeriesDataset(series_norm, window=window)
    split_idx = int(0.8 * len(ds))
    train_ds = torch.utils.data.Subset(ds, range(0, split_idx))
    val_ds = torch.utils.data.Subset(ds, range(split_idx, len(ds)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster()
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    mlflow.start_run(run_name=f"lstm_train_{int(time.time())}")
    mlflow.log_param("window", window)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, os.path.join(output_dir, "best_lstm.pth"))
    mlflow.log_metric("best_val_loss", best_val)
    mlflow.end_run()
    return os.path.join(output_dir, "best_lstm.pth")

def forecast_lstm(model_pth: str, history: List[float], steps: int = 10):
    ckpt = torch.load(model_pth, map_location=DEVICE)
    model = LSTMForecaster()
    model.load_state_dict(ckpt["state_dict"])
    mean = ckpt["mean"]
    std = ckpt["std"]
    model.to(DEVICE)
    model.eval()
    preds = []
    window = len(history)
    cur = np.array(history)
    # normalize
    cur_norm = (cur - mean) / (std + 1e-8)
    for _ in range(steps):
        x = torch.tensor(cur_norm[-window:]).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            out = model(x).cpu().numpy().item()
        # de-normalize
        val = out * std + mean
        preds.append(val)
        cur = np.append(cur, val)
        cur_norm = (cur - mean) / (std + 1e-8)
    return preds

# =========================
# AUTOENCODER FOR ANOMALY DETECTION (TABULAR)
# =========================
class TabularDataset(Dataset):
    def __init__(self, array: np.ndarray):
        self.array = array.astype(np.float32)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        return torch.tensor(self.array[idx])

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        # decoder
        decoder_layers = []
        hidden_dims_rev = list(reversed(hidden_dims))
        for h in hidden_dims_rev:
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.ReLU())
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

def train_autoencoder(csv_path: str, feature_cols: List[str], output_dir: str = "artifacts/ae", epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    ensure_dir(output_dir)
    df = load_tabular_csv(csv_path)
    validate_columns(df, feature_cols)
    X = df[feature_cols].fillna(0).values
    # scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    ds = TabularDataset(Xs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=Xs.shape[1])
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    mlflow.start_run(run_name=f"autoencoder_{int(time.time())}")
    mlflow.log_param("input_dim", Xs.shape[1])
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_ae.pth"))
    mlflow.log_metric("best_loss", best_loss)
    mlflow.end_run()
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_cols.joblib"))
    logger.info(f"Autoencoder trained, best_loss={best_loss}")
    return output_dir

def detect_anomalies_ae(model_dir: str, csv_path: str, threshold_multiplier: float = 3.0):
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    df = load_tabular_csv(csv_path)
    X = df[feature_cols].fillna(0).values
    Xs = scaler.transform(X)
    model = Autoencoder(input_dim=Xs.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_ae.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xs).to(DEVICE)
        recon = model(Xt).cpu().numpy()
    mse = np.mean((recon - Xs) ** 2, axis=1)
    # threshold as mean + k*std
    thresh = mse.mean() + threshold_multiplier * mse.std()
    anomalies = df[mse > thresh].copy()
    anomalies["reconstruction_error"] = mse[mse > thresh]
    return anomalies, thresh

# =========================
# DRIFT DETECTION & MONITORING
# =========================
def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10):
    """
    Compute PSIndex for numeric arrays. Lower is better; >0.25 often considered large shift.
    """
    expected = np.array(expected).ravel()
    actual = np.array(actual).ravel()
    # create bins based on expected
    hist_exp, bins = np.histogram(expected, bins=buckets)
    hist_act, _ = np.histogram(actual, bins=bins)
    # convert to percentage
    exp_perc = hist_exp / (hist_exp.sum() + 1e-8)
    act_perc = hist_act / (hist_act.sum() + 1e-8)
    psi = np.sum((exp_perc - act_perc) * np.log((exp_perc + 1e-8) / (act_perc + 1e-8)))
    return psi

def kolmogorov_smirnov_test(expected: np.ndarray, actual: np.ndarray):
    """
    Two-sample KS test. Returns statistic and p-value.
    """
    stat, p = stats.ks_2samp(expected.ravel(), actual.ravel())
    return stat, p

# =========================
# FASTAPI SERVING (PREDICTION + MANAGEMENT)
# =========================
app = FastAPI(title="Deep All-in-One Serving API")

# Request models
class ImagePredictRequest(BaseModel):
    model_path: str
    meta_path: str
    image_path: str

class TextPredictRequest(BaseModel):
    model_dir: str
    label_map: dict
    text: str

class TSForecastRequest(BaseModel):
    model_pth: str
    history: List[float]
    steps: int = 10

class AEAnomalyRequest(BaseModel):
    model_dir: str
    csv_path: str

@app.get("/")
def root():
    return {"message": "Deep Learning All-in-One API - healthy"}

@app.post("/predict/image")
def api_predict_image(req: ImagePredictRequest):
    try:
        label = predict_image(req.model_path, req.meta_path, req.image_path)
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/text")
def api_predict_text(req: TextPredictRequest):
    try:
        label = predict_text(req.model_dir, req.label_map, req.text)
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/ts")
def api_forecast_ts(req: TSForecastRequest):
    try:
        preds = forecast_lstm(req.model_pth, req.history, req.steps)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomaly/ae")
def api_anomaly_ae(req: AEAnomalyRequest):
    try:
        anomalies, thresh = detect_anomalies_ae(req.model_dir, req.csv_path)
        return {"anomalies_count": len(anomalies), "threshold": float(thresh)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Management endpoints (train triggers)
@app.post("/train/image")
def api_train_image(data_csv: str, epochs: int = 5):
    try:
        model_pth = train_image_classifier(data_csv, epochs=epochs)
        return {"model_pth": model_pth}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/text")
def api_train_text(data_csv: str, text_col: str = "text", label_col: str = "label"):
    try:
        out, label2id = train_text_classifier(data_csv, text_col, label_col)
        return {"model_dir": out, "label2id": label2id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/ts")
def api_train_ts(data_csv: str, value_col: str = "y", window: int = 24):
    try:
        model_pth = train_lstm_forecaster(data_csv, value_col, window)
        return {"model_pth": model_pth}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/ae")
def api_train_ae(data_csv: str, feature_cols: List[str]):
    try:
        out = train_autoencoder(data_csv, feature_cols)
        return {"ae_dir": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health-check endpoint with basic sanity tests
@app.get("/health")
def health_check():
    # Basic test: GPU availability, python packages
    info = {"device": str(DEVICE)}
    try:
        import torch, transformers
        info["torch_version"] = torch.__version__
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return info

# =========================
# CLI: Basic commands
# =========================
def main():
    parser = argparse.ArgumentParser(description="Deep Learning All-in-One CLI")
    subparsers = parser.add_subparsers(dest="cmd")

    # Train image
    p_img = subparsers.add_parser("train_image")
    p_img.add_argument("--data_csv", required=True)
    p_img.add_argument("--epochs", type=int, default=5)

    # Train text
    p_txt = subparsers.add_parser("train_text")
    p_txt.add_argument("--csv", required=True)
    p_txt.add_argument("--text_col", default="text")
    p_txt.add_argument("--label_col", default="label")
    p_txt.add_argument("--epochs", type=int, default=3)

    # Train ts
    p_ts = subparsers.add_parser("train_ts")
    p_ts.add_argument("--csv", required=True)
    p_ts.add_argument("--value_col", default="y")
    p_ts.add_argument("--epochs", type=int, default=10)

    # Train autoencoder
    p_ae = subparsers.add_parser("train_ae")
    p_ae.add_argument("--csv", required=True)
    p_ae.add_argument("--features", nargs='+', required=True)
    p_ae.add_argument("--epochs", type=int, default=20)

    # Serve
    p_serve = subparsers.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.cmd == "train_image":
        train_image_classifier(args.data_csv, epochs=args.epochs)
    elif args.cmd == "train_text":
        train_text_classifier(args.csv, args.text_col, args.label_col, epochs=args.epochs)
    elif args.cmd == "train_ts":
        train_lstm_forecaster(args.csv, value_col=args.value_col, epochs=args.epochs)
    elif args.cmd == "train_ae":
        train_autoencoder(args.csv, args.features, epochs=args.epochs)
    elif args.cmd == "serve":
        uvicorn.run("deep_learning_prod_all_in_one:app", host=args.host, port=args.port, reload=True)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
