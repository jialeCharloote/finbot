import torch
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model

import os
import gc
import numpy as np
import pandas as pd
import json
import random
import time
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

import torchvision
torchvision.disable_beta_transforms_warning()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"




# Check if CUDA is available and print GPU info
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = "cpu"
    print("WARNING: CUDA is not available, using CPU instead")
    print("Please check your SLURM GPU allocation")

# Set this at the beginning of your script
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

# For offline use
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Data processing
file_path = "/home/jialez/finetunemodel/balanced_news.json"

with open(file_path, "r") as f:
    articles = json.load(f)

news_data = Dataset.from_list(articles)
print(news_data[:5])

def process_news_data(example):
    """
    Process news data and generate Prompt-Response format for each ticker.
    """
    title = example["title"]
    description = example["description"]
    sentiment_map = {'negative': "Bearish", 'positive': "Bullish", 'neutral':'Neutral'}  # change the label
    label_map = {"Bearish":0, "Bullish":1,'Neutral':2}
    ticker = example["ticker"]
    new_sentiment = sentiment_map[example["sentiment"]]
    sentiment_reasoning = example["sentiment_reasoning"]

    prompt = f"""You are a financial analyst in a leading hedge fund. 
Analyze the sentiment of the following financial news for the given stock ticker step by step.

Title: "{title}"
Summary: "{description}"
Stock Ticker: {ticker}

Step 1: Identify key financial terms and their implications.
Step 2: Determine whether the news suggests market optimism, pessimism, or neutrality for this specific stock.
Step 3: Based on your analysis, classify the sentiment into one of the following categories:
- "Bullish": If the news suggests confidence, growth, or positive impact on this stock.
- "Bearish": If the news suggests decline, risks, or negative impact on this stock.
- "Neutral": If the news is ambiguous or does not convey strong sentiment.

Finally, **return only** the final result in valid JSON format, with the structure:
{{
  "ticker": "{ticker}",
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "sentiment_reasoning": "Provide a brief explanation of the sentiment analysis."
}}

Do not include any extra text or explanations outside the JSON.
### Response:
(Return a valid JSON format only. Do not repeat the prompt.)
"""
    response = json.dumps({
            "ticker": ticker,
            "sentiment": new_sentiment,
            "sentiment_reasoning": sentiment_reasoning
        })

    return [{"prompt": prompt, "response": response, "labels":label_map[new_sentiment], "ground_truth_sentiment":new_sentiment}]

# Convert `news_data` into multiple samples corresponding to `prompt-response`
flattened_data = []
for example in news_data:
    processed = process_news_data(example)
    if processed:
        flattened_data.extend(processed)

news_dataset = Dataset.from_list(flattened_data)

del flattened_data, processed, example

# Stratified sampling
df = pd.DataFrame(news_dataset)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2,  # 80% training, 20% test
    stratify=df["ground_truth_sentiment"],  # Stratified by sentiment
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

print("Train set distribution:", Counter(train_dataset["ground_truth_sentiment"]))
print("Test set distribution:", Counter(test_dataset["ground_truth_sentiment"]))
del df, train_df, test_df

# Prepare text for training
def concat_prompt_response(example):
    return {"text": example["prompt"] + example["response"]}

train_dataset = train_dataset.map(concat_prompt_response)
test_dataset = test_dataset.map(concat_prompt_response)

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Load model and tokenizer from local paths
model_path = "/home/jialez/finetunemodel/hf_cache/ds_model"
tokenizer_path = "/home/jialez/finetunemodel/hf_cache/ds_model"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True,local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.config.use_cache = False

# Configure LoRA for GPT-2
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # GPT2-specific target modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# Tokenize datasets
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

train_dataset = train_dataset.map(tokenize_function, 
                                 batched=True, num_proc=4, 
                                 remove_columns=["prompt", "response", "text"])
test_dataset = test_dataset.map(tokenize_function, 
                               batched=True, num_proc=4, 
                               remove_columns=["prompt", "response", "text"])

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# Metrics calculation
import math
from torch.nn.utils.rnn import pad_sequence

# Label mappings
label2id = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
id2label = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

def parse_json_output(output_text):
    """
    Extracts sentiment from the model's JSON response.
    """
    json_match = re.search(r'\{.*?\}', output_text, re.DOTALL)
    if json_match:
        extracted_json = json_match.group(0)
        try:
            parsed = json.loads(extracted_json)
            return parsed.get("sentiment", "Neutral")
        except json.JSONDecodeError:
            print(f"⚠️ JSON Decode Error in output: {output_text}")
    return "Neutral"  # Default if parsing fails

def generate_sentiments_in_batches(dataset, model, tokenizer, batch_size=8, max_new_tokens=100):
    """
    Processes dataset in batches and extracts sentiment predictions from model responses.
    """
    device = model.device
    all_pred_sentiments = []

    for start_idx in range(0, len(dataset), batch_size):
        batch_dict = dataset[start_idx: start_idx + batch_size]
        input_ids_list = batch_dict["input_ids"]
        attn_mask_list = batch_dict["attention_mask"]

        # Convert to tensors and pad
        input_ids_tensors = [torch.tensor(seq, dtype=torch.long) for seq in input_ids_list]
        attn_mask_tensors = [torch.tensor(seq, dtype=torch.long) for seq in attn_mask_list]

        input_ids_padded = pad_sequence(input_ids_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attn_mask_padded = pad_sequence(attn_mask_tensors, batch_first=True, padding_value=0).to(device)

        input_length = input_ids_padded.shape[1]

        # Generate model response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_padded,
                attention_mask=attn_mask_padded,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                do_sample=False,
                num_return_sequences=1,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Extract sentiment from model outputs
        for out_seq in outputs:
            generated_text = tokenizer.decode(out_seq[input_length:], skip_special_tokens=True).strip()
            pred_sentiment = parse_json_output(generated_text)
            all_pred_sentiments.append(pred_sentiment)

    return all_pred_sentiments

def compute_metrics(eval_preds):
    """
    Computes classification accuracy, F1 score, and perplexity.
    """
    logits, labels = eval_preds

    # Convert logits to tensor
    logits = torch.tensor(logits, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    # Compute language modeling loss (Cross Entropy)
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    )
    perplexity = math.exp(loss.item())

    # Generate model-based predictions for evaluation dataset
    all_pred_sentiments = generate_sentiments_in_batches(
        test_dataset, model, tokenizer,
        batch_size=2, max_new_tokens=100
    )

    # Extract true sentiments
    all_true_sentiments = [row["ground_truth_sentiment"] for row in test_dataset]

    # Convert to numerical format
    pred_sent_ids = [label2id.get(ps, 2) for ps in all_pred_sentiments]  # Default "Neutral" = 2
    true_sent_ids = [label2id[ts] for ts in all_true_sentiments]

    # Compute classification metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(true_sent_ids, pred_sent_ids)
    f1 = f1_score(true_sent_ids, pred_sent_ids, average="macro")

    return {
        "eval_loss": loss.item(),
        "perplexity": perplexity,
        "classification_accuracy": accuracy,
        "classification_f1_macro": f1
    }

# Training configuration
training_args = TrainingArguments(
    output_dir="/home/jialez/finetunemodel/finetuned-ds",
    run_name="finetuned-gpt2-run-1",
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine_with_restarts",
    learning_rate=5e-5,
    gradient_checkpointing=False,
    
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=10,
    warmup_ratio=0.1,
    
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=1000,
    dataloader_pin_memory=True, 
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    label_names=["labels"],
    max_grad_norm=5.0,
    fp16=True,
    dataloader_num_workers=4,
    push_to_hub=False,
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Save the model

trainer.save_model("/home/jialez/finetunemodel/finetuned-ds")
tokenizer.save_pretrained("/home/jialez/finetunemodel/finetuned-ds")
print("✅ Training completed. Successfully saved model")

loghistory=pd.DataFrame(trainer.state.log_history)
loghistory.to_csv("/home/jialez/finetunemodel/finetuned-ds/loghistory.csv", index=False)

# Create zip file for download
import shutil
shutil.make_archive("finetuned-ds", 'zip', "/home/jialez/finetunemodel/finetuned-ds")
print("✅ finetuned-ds.zip is available to download")
