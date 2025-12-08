import subprocess
from pathlib import Path
from pprint import pprint
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorForTokenClassification
)
from datasets import Dataset, concatenate_datasets
import pickle

# --- Paths & model ---
VARD_DIR = "../my_macberth/lib/VARD2.5.4"
JARS = [
    f"{VARD_DIR}/vardstdin.jar",
    f"{VARD_DIR}/model.jar",
    f"{VARD_DIR}/clui.jar"
]
CLASSPATH = ":".join(JARS)  # Linux/Mac classpath separator
VARD_MAIN_CLASS = "vardstdin.VARDMain"

RAW_TEXTS_DIR = Path("./texts")
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

FINE_TUNED_DIR = Path("./our_fine_tuned_macberth")

MODEL_NAME = "emanjavacas/MacBERTh"

# --- Initialize MacBERTh ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if FINE_TUNED_DIR.exists():
    print("Loading fine-tuned MacBERTh from checkpoint...")
    model = AutoModelForTokenClassification.from_pretrained(FINE_TUNED_DIR)
else:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, device=-1)

# --- Function to normalize text using VARD2 via STDIN/STDOUT ---
def normalize_with_vard(text: str, cache_file: Path) -> str:
    """Normalize text using VARD2, with caching."""
    if cache_file.exists():
        return cache_file.read_text()
    
    try:
        proc = subprocess.Popen(
            ["java", "-cp", CLASSPATH, VARD_MAIN_CLASS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(input=text)
        if proc.returncode != 0:
            print("VARD2 failed!")
            print("stderr:", stderr)
            return text  # fallback
        normalized = stdout.strip()
        cache_file.write_text(normalized)
        return normalized
    except Exception as e:
        print("Exception while running VARD2:", e)
        return text  # fallback

# --- Incremental fine-tuning ---
def incremental_fine_tune(new_texts, new_labels):
    """Fine-tune MacBERTh incrementally with new texts and labels."""
    # Load previous dataset if exists
    dataset_cache_file = CACHE_DIR / "dataset.pkl"
    if dataset_cache_file.exists():
        with dataset_cache_file.open("rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} previously fine-tuned examples.")
    else:
        dataset = None

    # Tokenize new texts
    tokenized_inputs = tokenizer(
        new_texts, truncation=True, padding=True, is_split_into_words=True
    )
    tokenized_labels = []
    for i, lbl in enumerate(new_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        token_labels = []
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)
            else:
                token_labels.append(lbl[word_id])
        tokenized_labels.append(token_labels)

    new_dataset = Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels
    })

    if dataset is not None:
        dataset = concatenate_datasets([dataset, new_dataset])
    else:
        dataset = new_dataset

    # Save dataset cache
    with dataset_cache_file.open("wb") as f:
        pickle.dump(dataset, f)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(FINE_TUNED_DIR),
        overwrite_output_dir=False,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=1
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(FINE_TUNED_DIR)
    tokenizer.save_pretrained(FINE_TUNED_DIR)
    print("Incrementally fine-tuned MacBERTh saved.")


# --- Combined pipeline ---
def normalize_and_classify(text_file: Path):
    text = text_file.read_text()
    cache_file = CACHE_DIR / f"{text_file.stem}_normalized.txt"
    normalized_text = normalize_with_vard(text, cache_file)
    classified = nlp(normalized_text)
    return {"original": text, "normalized": normalized_text, "classified": classified}


# --- Example usage ---
if __name__ == "__main__":
    # Normalize and classify all texts in RAW_TEXTS_DIR
    for txt_file in RAW_TEXTS_DIR.glob("*.txt"):
        result = normalize_and_classify(txt_file)
        pprint(result)

    # Example: incremental fine-tuning
    # Replace these with your actual tokenized texts and labels
    # sample_texts = [["Thys", "is", "a", "pamphlet", "sentence", "."]]
    # sample_labels = [[1, 1, 1, 1, 1, 1]]
    # incremental_fine_tune(sample_texts, sample_labels)
