"""
finetune_text_classifier.py
---------------------------
End-to-End Fine-Tuning Example using Hugging Face Transformers

Use case: Text classification (e.g. sentiment analysis, category prediction)

Steps:
1️⃣ Load dataset (CSV with 'text' and 'label' columns)
2️⃣ Encode text + labels
3️⃣ Fine-tune a pretrained model (DistilBERT)
4️⃣ Save model and tokenizer
5️⃣ Load fine-tuned model and make predictions
"""

# ===============================
# 🧩 Imports
# ===============================
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ===============================
# ⚙️ Configuration
# ===============================
MODEL_NAME = "distilbert-base-uncased"     # Pretrained model checkpoint
OUTPUT_DIR = "finetuned_model"             # Where to save fine-tuned model
EPOCHS = 3                                 # You can tune this
BATCH_SIZE = 16
LR = 2e-5

# ===============================
# 📥 1️⃣ Load and Prepare Dataset
# ===============================
def load_data(csv_path="data.csv"):
    """
    CSV must have columns: 'text' and 'label'
    Example:
        text,label
        I love this movie!,positive
        Worst film ever,bad
    """
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded data: {df.shape[0]} samples")
    # Encode labels numerically
    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)
    return df, label2id, id2label

# ===============================
# 🧠 2️⃣ Tokenization
# ===============================
def tokenize_data(df, tokenizer):
    hf_dataset = Dataset.from_pandas(df[["text", "label_id"]])
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    tokenized = hf_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

# ===============================
# 🏋️‍♂️ 3️⃣ Fine-Tuning Function
# ===============================
def fine_tune(csv_path="data.csv"):
    df, label2id, id2label = load_data(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize_data(df, tokenizer)
    train_test = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds = train_test["train"]
    test_ds = train_test["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="logs",
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    print("✅ Training complete!")

    # Save everything
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Model saved to {OUTPUT_DIR}")

    # Evaluate
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))

    return OUTPUT_DIR, label2id, id2label

# ===============================
# 🔍 4️⃣ Load & Predict
# ===============================
def predict(texts: list, model_dir: str = OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    # Convert ids back to labels if available
    id2label = model.config.id2label
    labels = [id2label[p] for p in preds]
    return labels

# ===============================
# ▶️ Main
# ===============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--csv", default="data.csv")
    parser.add_argument("--text", nargs="+", help="Texts to predict")
    args = parser.parse_args()

    if args.mode == "train":
        fine_tune(args.csv)
    elif args.mode == "predict":
        if not args.text:
            print("❌ Please provide text(s) to predict with --text")
        else:
            preds = predict(args.text)
            for t, p in zip(args.text, preds):
                print(f"🗣️ '{t}' → {p}")




# torch
# transformers
# datasets
# pandas
# numpy
# scikit-learn


# data.csv

# text,label
# I love this product!,positive
# This is the worst purchase I've made,negative
# Absolutely amazing quality,positive
# Terrible support and slow delivery,negative


# python finetune_text_classifier.py train --csv data.csv



# 💬 Run Predictions
# python finetune_text_classifier.py predict --text "I enjoyed the movie" "This was awful"


# ✅ Output:

# 🗣️ 'I enjoyed the movie' → positive
# 🗣️ 'This was awful' → negative


















# lora finetune


"""
finetune_lora_llama.py
----------------------
End-to-End LoRA fine-tuning example for LLaMA / Mistral / Gemma models
using Hugging Face Transformers + PEFT.

✅ Efficient: updates <1% of model parameters
✅ Works on a single GPU (8–16 GB)
✅ Compatible with Hugging Face Hub

Steps:
1️⃣ Load base model & tokenizer
2️⃣ Prepare your dataset (CSV, JSONL, etc.)
3️⃣ Apply LoRA adapters
4️⃣ Fine-tune
5️⃣ Save / merge adapters
6️⃣ Inference

Requirements:
    transformers
    peft
    datasets
    bitsandbytes
    accelerate
    torch
    pandas
"""

# ===============================
# 📦 Imports
# ===============================
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ===============================
# ⚙️ Configuration
# ===============================
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "lora_llama_finetuned"
DATA_PATH = "train_data.csv"             # your dataset file
TEXT_COLUMN = "text"                     # column to fine-tune on
EPOCHS = 2
BATCH_SIZE = 2
LR = 2e-4
LORA_RANK = 8

# ===============================
# 🧹 1️⃣ Load and prepare dataset
# ===============================
def load_dataset(data_path=DATA_PATH, text_col=TEXT_COLUMN):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[text_col])
    print(f"✅ Loaded {len(df)} samples from {data_path}")
    return Dataset.from_pandas(df)

# ===============================
# 🔧 2️⃣ Tokenization
# ===============================
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=512)
    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# ===============================
# 🧠 3️⃣ LoRA fine-tuning
# ===============================
def finetune_lora():
    # Load base model in 8-bit to save VRAM
    print("⏳ Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # key layers in transformer blocks
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap base model with LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize dataset
    dataset = load_dataset(DATA_PATH, TEXT_COLUMN)
    tokenized = tokenize_dataset(dataset, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning complete!")

    # Save adapter weights only (small size)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Saved LoRA adapters to {OUTPUT_DIR}")

# ===============================
# 🔍 4️⃣ Inference (using LoRA adapters)
# ===============================
def generate_response(prompt: str, model_dir: str = OUTPUT_DIR):
    print("⏳ Loading model + LoRA adapters...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🧠 Model Output:")
    print(response)
    return response

# ===============================
# ▶️ Main entry point
# ===============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "inference"])
    parser.add_argument("--prompt", default="Explain LoRA fine-tuning in simple terms.")
    args = parser.parse_args()

    if args.mode == "train":
        finetune_lora()
    elif args.mode == "inference":
        generate_response(args.prompt)

# torch
# transformers>=4.36
# peft>=0.7.0
# bitsandbytes>=0.41
# datasets
# accelerate
# pandas

# 🧪 Example Dataset (train_data.csv)

# You can fine-tune on any conversational or instruction-style data.
# Example:

# text
# User: Explain what LoRA is.
# Assistant: LoRA is a parameter-efficient fine-tuning method that updates small adapter matrices instead of the full model.
# User: What’s the benefit of LoRA?
# Assistant: It reduces VRAM and training time by fine-tuning <1% of parameters.
# 

# You can also prepare Alpaca-style data with JSON if you want to train for instruction following.

# ▶️ Fine-tune on your data
# python finetune_lora_llama.py train


# ✅ Output:

# Adapter weights (~150 MB) saved in lora_llama_finetuned/.

# 💬 Inference (use fine-tuned model)
# python finetune_lora_llama.py inference --prompt "Explain LoRA in 2 lines."


# ✅ Output:

# 🧠 Model Output:
# LoRA fine-tunes large models efficiently by adding small trainable adapters to specific layers, saving memory and time.
