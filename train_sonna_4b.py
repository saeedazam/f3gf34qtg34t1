"""
train_sonna_4b.py
Kaggle-optimized training script for Sonna-4B using Unsloth.
Loads local SFT datasets and high-quality HuggingFace datasets.
"""
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets
import os

# 1. Configuration - Sonna-1.1 identity
SYSTEM_PROMPT = (
    "You are Sonna 1.1, designed by Grand Guard, made to run locally. "
    "You are a direct, friendly, and capable AI assistant. "
    "You are great at coding, answering questions, and going straight to the point. "
    "When asked who you are or about your identity, always respond with: "
    "'I am Sonna 1.1 designed by Grand Guard made to run Locally!' "
    "You are not ChatGPT, not Gemma, not Qwen, not made by any big tech company. "
    "You are Sonna 1.1 by Grand Guard, period. "
    "Be helpful, concise, and clear. No fluff."
)

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 4096 # Increased for reasoning datasets
dtype = None
load_in_4bit = True

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters - Increased rank for coding excellence
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Increased from 32 for deeper logic retention
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Data Preparation
print("\n[Sonna] Loading local datasets...")
identity_dataset = load_dataset("json", data_files="sonna_identity_mcp.jsonl", split="train")
sft_dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")
mcp_ui_dataset = load_dataset("json", data_files="mcp_ui_data.jsonl", split="train")
adv_cap_dataset = load_dataset("json", data_files="advanced_capabilities.jsonl", split="train")
claude_dataset = load_dataset("json", data_files="claude_examples.jsonl", split="train")

print("\n[Sonna] Loading HuggingFace SOTA datasets...")
# FineTome-100k for high-quality instruction following
finetome_dataset = load_dataset("mlabonne/FineTome-100k", split="train[:15000]")

# OpenThoughts-114k for DeepSeek-R1 style reasoning & complex coding
reasoning_dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train[:10000]")

# CodeFeedback for superior programming skills
coding_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:10000]")

# List of common conversational filler to strip
CONCISE_FILLERS = ["Certainly!", "Sure,", "I can help with that.", "Of course!", "Here is", "I'd be happy to", "To answer your question,"]

def strip_fillers(text):
    for filler in CONCISE_FILLERS:
        if text.startswith(filler):
            text = text[len(filler):].strip()
    return text

def format_local_prompts(examples):
    batch_size = len(next(iter(examples.values())))
    prompts = examples.get("prompt", examples.get("instruction", [""] * batch_size))
    completions = examples.get("completion", examples.get("output", [""] * batch_size))
    
    # Handle cases where .get returned []
    if len(prompts) == 0: prompts = [""] * batch_size
    if len(completions) == 0: completions = [""] * batch_size
    
    texts = []
    for p, c in zip(prompts, completions):
        clean_p = str(p).replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "").strip()
        clean_c = strip_fillers(str(c).replace("<|assistant|>", "").replace("<|end|>", "").strip())
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{clean_p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_c}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_finetome(examples):
    conversations = examples["conversations"]
    texts = []
    for conv in conversations:
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        for turn in conv:
            role = "user" if turn["from"] == "human" else "assistant"
            val = strip_fillers(turn["value"]) if role == "assistant" else turn["value"]
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{val}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_coding(examples):
    batch_size = len(next(iter(examples.values())))
    query = examples.get("query", examples.get("instruction", examples.get("prompt", [""] * batch_size)))
    answer = examples.get("answer", examples.get("output", examples.get("completion", [""] * batch_size)))
    
    if len(query) == 0: query = [""] * batch_size
    if len(answer) == 0: answer = [""] * batch_size

    texts = []
    for q, a in zip(query, answer):
        clean_a = strip_fillers(str(a))
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{str(q)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_a}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_reasoning(examples):
    batch_size = len(next(iter(examples.values())))
    instruction = examples.get("instruction", examples.get("prompt", examples.get("question", [""] * batch_size)))
    thought = examples.get("thought", examples.get("reasoning", [""] * batch_size))
    output = examples.get("output", examples.get("answer", examples.get("completion", [""] * batch_size)))
    
    if len(instruction) == 0: instruction = [""] * batch_size
    if len(thought) == 0: thought = [""] * batch_size
    if len(output) == 0: output = [""] * batch_size

    texts = []
    for i, t, o in zip(instruction, thought, output):
        clean_o = strip_fillers(str(o))
        if t and str(t).strip():
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{str(i)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thought>\n{str(t)}\n</thought>\n\n{clean_o}<|eot_id|>"
        else:
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{str(i)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_o}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }


# Define columns to remove to keep dataset clean
def get_cols_to_remove(dataset):
    return [col for col in dataset.column_names if col != "text"]

identity_dataset = identity_dataset.map(format_local_prompts, batched=True)
identity_dataset = identity_dataset.remove_columns(get_cols_to_remove(identity_dataset))

sft_dataset = sft_dataset.map(format_local_prompts, batched=True)
sft_dataset = sft_dataset.remove_columns(get_cols_to_remove(sft_dataset))

mcp_ui_dataset = mcp_ui_dataset.map(format_local_prompts, batched=True)
mcp_ui_dataset = mcp_ui_dataset.remove_columns(get_cols_to_remove(mcp_ui_dataset))

adv_cap_dataset = adv_cap_dataset.map(format_local_prompts, batched=True)
adv_cap_dataset = adv_cap_dataset.remove_columns(get_cols_to_remove(adv_cap_dataset))

claude_dataset = claude_dataset.map(format_local_prompts, batched=True)
claude_dataset = claude_dataset.remove_columns(get_cols_to_remove(claude_dataset))

finetome_dataset = finetome_dataset.map(format_finetome, batched=True)
finetome_dataset = finetome_dataset.remove_columns(get_cols_to_remove(finetome_dataset))

reasoning_dataset = reasoning_dataset.map(format_reasoning, batched=True)
reasoning_dataset = reasoning_dataset.remove_columns(get_cols_to_remove(reasoning_dataset))

coding_dataset = coding_dataset.map(format_coding, batched=True)
coding_dataset = coding_dataset.remove_columns(get_cols_to_remove(coding_dataset))

# Combine ALL datasets (now all have exactly ONE column: 'text')
dataset = concatenate_datasets([
    identity_dataset, identity_dataset, 
    mcp_ui_dataset, mcp_ui_dataset,    
    coding_dataset, coding_dataset,    
    adv_cap_dataset,
    claude_dataset,
    sft_dataset,
    finetome_dataset,
    reasoning_dataset
])
dataset = dataset.shuffle(seed=3407)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 5. Final Tokenizer & Dataset Preparation
# Llama 3.2 doesn't have a pad token by default; we MUST set one.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Standard for training

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False, # We use dynamic padding in the collator
    )

print("\n[Sonna] Tokenizing dataset for training...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=2,
)
# 6. Training using the standard Trainer for maximum stability
trainer = Trainer(
    model = model,
    train_dataset = tokenized_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 1500, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        # DEEP PATCH: These three flags together stop the .mean() crash
        average_tokens_across_devices = False,
        eval_strategy = "no",
        do_eval = False,
    ),
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\n[Sonna] Starting training loop...")
trainer.train()

# 6. Save & Export
print("\n[Sonna] Saving LoRA adapters...")
model.save_pretrained("sonna-4b-lora")
tokenizer.save_pretrained("sonna-4b-lora")

print("\n[Sonna] Exporting to GGUF (Q4_K_M)...")
model.save_pretrained_gguf("sonna-4b-gguf", tokenizer, quantization_method = "q4_k_m")
print("\n[Sonna] DONE.")
