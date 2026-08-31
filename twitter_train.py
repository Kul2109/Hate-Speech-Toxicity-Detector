import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER RoBERTa V3 TRAINING
# CPU OPTIMIZED
# ============================================================

MODEL_NAME = "roberta-base"

TRAIN_FILE = "datasets/twitter/train.csv"
VAL_FILE = "datasets/twitter/validation.csv"
TEST_FILE = "datasets/twitter/test.csv"

# IMPORTANT:
# Keep v2 safe. New model will be saved here.
OUTPUT_DIR = "models/twitter_roberta_v3"

# Faster than the previous 96
MAX_LENGTH = 64

LABEL_NAMES = [
    "Hate Speech",
    "Offensive Language",
    "Neither",
]

NUM_LABELS = 3


# ============================================================
# START
# ============================================================

print("=" * 70)
print("Hate Speech Toxicity Detector - TWITTER RoBERTa V3 TRAINING")
print("=" * 70)

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print("Training device: CPU")
    print("CPU optimized configuration enabled.")


# ============================================================
# CHECK DATASET FILES
# ============================================================

print("\nChecking dataset files...")

required_files = [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nDataset file not found:\n{file_path}"
        )

    print(
        "FOUND:",
        file_path
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n")
print("=" * 70)
print("LOADING TWITTER DATASETS")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


# ------------------------------------------------------------
# Keep required columns only
# ------------------------------------------------------------

train_df = train_df[
    ["tweet", "class"]
].copy()

val_df = val_df[
    ["tweet", "class"]
].copy()

test_df = test_df[
    ["tweet", "class"]
].copy()


# ------------------------------------------------------------
# Remove missing values
# ------------------------------------------------------------

train_df = train_df.dropna(
    subset=["tweet", "class"]
)

val_df = val_df.dropna(
    subset=["tweet", "class"]
)

test_df = test_df.dropna(
    subset=["tweet", "class"]
)


# ------------------------------------------------------------
# Convert labels to integers
# ------------------------------------------------------------

train_df["class"] = train_df["class"].astype(int)

val_df["class"] = val_df["class"].astype(int)

test_df["class"] = test_df["class"].astype(int)


# ------------------------------------------------------------
# Convert tweet to string
# ------------------------------------------------------------

train_df["tweet"] = train_df["tweet"].astype(str)

val_df["tweet"] = val_df["tweet"].astype(str)

test_df["tweet"] = test_df["tweet"].astype(str)


# ------------------------------------------------------------
# Rename class -> labels
# ------------------------------------------------------------

train_df = train_df.rename(
    columns={
        "class": "labels"
    }
)

val_df = val_df.rename(
    columns={
        "class": "labels"
    }
)

test_df = test_df.rename(
    columns={
        "class": "labels"
    }
)


print("\nDataset sizes:")

print(
    "Training samples  :",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)

print(
    "Testing samples   :",
    len(test_df)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = (
    train_df["labels"]
    .value_counts()
    .sort_index()
)

for label in range(NUM_LABELS):

    count = int(
        class_counts.get(
            label,
            0
        )
    )

    print(
        f"{label} - {LABEL_NAMES[label]}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

total_samples = len(
    train_df
)

weights = []

for label in range(NUM_LABELS):

    count = int(
        class_counts.get(
            label,
            1
        )
    )

    weight = (
        total_samples /
        (
            NUM_LABELS *
            count
        )
    )

    weights.append(
        weight
    )


class_weights = torch.tensor(
    weights,
    dtype=torch.float
)


for label in range(NUM_LABELS):

    print(
        f"{LABEL_NAMES[label]:25s}: "
        f"{class_weights[label].item():.4f}"
    )


# ============================================================
# HUGGING FACE DATASETS
# ============================================================

print("\nCreating Hugging Face datasets...")

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\n")
print("=" * 70)
print("LOADING RoBERTa TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["tweet"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ============================================================
# TOKENIZE
# ============================================================

print("\nTokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training data"
)


print("Tokenizing validation dataset...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation data"
)


print("Tokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test data"
)


# ============================================================
# MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING RoBERTa MODEL")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_LABELS,

    id2label={
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither",
    },

    label2id={
        "Hate Speech": 0,
        "Offensive Language": 1,
        "Neither": 2,
    },

)


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        labels = inputs.pop(
            "labels"
        )

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        weights = class_weights.to(
            logits.device
        )

        loss_function = (
            torch.nn.CrossEntropyLoss(
                weight=weights
            )
        )

        loss = loss_function(
            logits,
            labels
        )

        if return_outputs:

            return (
                loss,
                outputs
            )

        return loss


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    predictions = (
        eval_prediction.predictions
    )

    labels = (
        eval_prediction.label_ids
    )


    predicted_labels = np.argmax(
        predictions,
        axis=1
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )


    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )


    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )


    return {

        "accuracy": accuracy,

        "precision_macro": precision_macro,

        "recall_macro": recall_macro,

        "f1_macro": f1_macro,

        "hate_precision": precision[0],

        "hate_recall": recall[0],

        "hate_f1": f1[0],

        "offensive_precision": precision[1],

        "offensive_recall": recall[1],

        "offensive_f1": f1[1],

        "neither_precision": precision[2],

        "neither_recall": recall[2],

        "neither_f1": f1[2],

    }


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    "Model          :",
    MODEL_NAME
)

print(
    "Max length     :",
    MAX_LENGTH
)

print(
    "Epochs         :",
    1
)

print(
    "Train batch    :",
    8
)

print(
    "Eval batch     :",
    8
)

print(
    "Learning rate  :",
    "2e-5"
)

print(
    "Output         :",
    OUTPUT_DIR
)


training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Faster CPU configuration
    # --------------------------------------------------------

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    gradient_accumulation_steps=1,

    # --------------------------------------------------------
    # Start with one epoch
    # --------------------------------------------------------

    num_train_epochs=1,

    learning_rate=2e-5,

    weight_decay=0.01,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=200,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=1,

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",

    greater_is_better=True,

    # --------------------------------------------------------
    # No external reporting
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    fp16=False,

    bf16=False,

    dataloader_num_workers=0,

)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,

)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING RoBERTa V3 TRAINING")
print("=" * 70)

print(
    "\nThis is the FAST CPU TEST VERSION."
)

print(
    "Training for 1 epoch first."
)

print(
    "Your existing twitter_roberta_v2 "
    "will NOT be changed."
)

print(
    "\nPlease do not close this terminal."
)


trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)


for key, value in validation_results.items():

    if isinstance(
        value,
        (
            float,
            np.floating
        )
    ):

        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)


for key, value in test_results.items():

    if isinstance(
        value,
        (
            float,
            np.floating
        )
    ):

        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


predictions = trainer.predict(
    test_dataset
)


predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)


true_labels = (
    predictions.label_ids
)


print(
    classification_report(

        true_labels,

        predicted_labels,

        labels=[
            0,
            1,
            2
        ],

        target_names=LABEL_NAMES,

        digits=4,

        zero_division=0,

    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(

    true_labels,

    predicted_labels,

    labels=[
        0,
        1,
        2
    ]

)


cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual Hate Speech",
        "Actual Offensive",
        "Actual Neither",
    ],

    columns=[
        "Predicted Hate Speech",
        "Predicted Offensive",
        "Predicted Neither",
    ]

)


print(
    cm_df
)


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANT PER-CLASS RESULTS")
print("=" * 70)


report = classification_report(

    true_labels,

    predicted_labels,

    labels=[
        0,
        1,
        2
    ],

    target_names=LABEL_NAMES,

    output_dict=True,

    zero_division=0,

)


for label_name in LABEL_NAMES:

    result = report[label_name]

    print(
        f"\n{label_name}"
    )

    print(
        f"  Precision : {result['precision']:.4f}"
    )

    print(
        f"  Recall    : {result['recall']:.4f}"
    )

    print(
        f"  F1 Score  : {result['f1-score']:.4f}"
    )

    print(
        f"  Samples   : {int(result['support'])}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n")
print("=" * 70)
print("SAVING RoBERTa V3 MODEL")
print("=" * 70)


trainer.save_model(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    "\nModel saved at:"
)


print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TWITTER RoBERTa V3 TRAINING COMPLETED")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "Do NOT change twitter_service.py yet."
)

print(
    "First test the new v3 model."
)

print(
    "If v3 performs better, we will connect it "
    "to The Hate Speech Toxicity Detector."
)

print(
    "\nExisting v2 model remains untouched:"
)

print(
    os.path.abspath(
        "models/twitter_roberta_v2"
    )
)

print(
    "\nNew v3 model:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)