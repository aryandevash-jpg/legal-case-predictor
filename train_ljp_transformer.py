import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

# -----------------------------
# 1. Custom Dataset
# -----------------------------
class LJPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding done by DataCollator
        )
        encodings["labels"] = label
        return encodings


# -----------------------------
# 2. Metrics
# -----------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


# -----------------------------
# 3. Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train LJP Transformer model")

    parser.add_argument(
        "--data_path",
        type=str,
        default="Realistic_LJP_Facts.csv",
        help="Path to CSV dataset with columns: text, label, split",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpaueb/legal-bert-base-uncased",
        help="Base transformer model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ljp_legalbert_model",
        help="Where to save model and tokenizer",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for LR scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


# -----------------------------
# 4. Main training function
# -----------------------------
def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 4.1 Load CSV
    df = pd.read_csv(args.data_path)

    # Basic sanity check
    required_cols = {"text", "label", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Split data
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    print(f"Train size: {len(train_df)}")
    print(f"Dev size:   {len(dev_df)}")
    print(f"Test size:  {len(test_df)}")

    # 4.2 Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # binary classification: 0/1
    )

    # 4.3 Create Dataset objects
    train_dataset = LJPDataset(
        texts=train_df["text"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dev_dataset = LJPDataset(
        texts=dev_df["text"],
        labels=dev_df["label"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_dataset = LJPDataset(
        texts=test_df["text"],
        labels=test_df["label"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # 4.4 Data collator (dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4.5 Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",   # eval after each epoch on dev
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),   # use mixed precision if GPU
    )

    # 4.6 Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4.7 Train
    trainer.train()

    # 4.8 Evaluate on dev and test
    print("Evaluating on dev set...")
    dev_metrics = trainer.evaluate(eval_dataset=dev_dataset)
    print("Dev metrics:", dev_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Test metrics:", test_metrics)

    # 4.9 Save final model + tokenizer
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete ✅")


if __name__ == "__main__":
    main()
