import os
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./flan_t5_lora_multitask"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

print("Loading dataset...")
df = pd.read_csv("merged_preprocessed_dataset.csv")


df = df.dropna(subset=["target_text", "input_text"]).reset_index(drop=True)


TASK_PREFIXES = {
    "gec": "fix grammar: ",
    "paraphrase": "paraphrase: ",
    "summarization": "summarize: ",
}


def apply_prefix(row):
    prefix = TASK_PREFIXES.get(str(row["task"]).lower(), "process: ")
    return prefix + str(row["input_text"]).strip()


df["formatted_input"] = df.apply(apply_prefix, axis=1)


train_df, val_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["task"]
)

train_dataset = Dataset.from_pandas(
    train_df[["formatted_input", "target_text"]].reset_index(drop=True)
)
val_dataset = Dataset.from_pandas(
    val_df[["formatted_input", "target_text"]].reset_index(drop=True)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["formatted_input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_val = val_dataset.map(
    preprocess_function, batched=True, remove_columns=val_dataset.column_names
)

print("Loading model and applying LoRA...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,  
    lora_alpha=32, 
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, pad_to_multiple_of=8
)

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]


    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    rouge_result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    bleu_result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
    )

    return {
        "rouge1": round(rouge_result["rouge1"] * 100, 2),
        "rougeL": round(rouge_result["rougeL"] * 100, 2),
        "bleu": round(bleu_result["bleu"] * 100, 2),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2, 
    gradient_checkpointing=True,
    num_train_epochs=3,
    predict_with_generate=True,
    bf16=True, 
    fp16=False,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=1,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_steps=100,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print("Starting LoRA Fine-Tuning...")
    trainer.train()

    print("\nEvaluating Model Metrics...")
    eval_results = trainer.evaluate()
    print("Final Validation Metrics:", eval_results)

    print("\nSaving LoRA Adapters...")
    model.save_pretrained(f"{OUTPUT_DIR}/best_lora_weights")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_lora_weights")
    print("Training complete!")