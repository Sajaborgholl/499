import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

# Paths
dataset_path = "formatted_dataset.json"
output_dir = "lora-deepseek-output"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base", use_fast=True)

# Tokenization function


def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True,
                       padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# BitsAndBytes 4-bit quantization config
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-base"
)


# Prepare model for LoRA
model = prepare_model_for_kbit_training(base_model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir=os.path.join(output_dir, "logs"),
    save_total_limit=1,
    save_strategy="epoch"
)

# Train the model
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args
)
trainer.train()

# Save model and tokenizer
model.save_pretrained("lora-deepseek-aggressive")
tokenizer.save_pretrained("lora-deepseek-aggressive")

# Export models for use in other files
AGGRESSIVE_MODEL = model
BASE_MODEL = base_model
TOKENIZER = tokenizer
