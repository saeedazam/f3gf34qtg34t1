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

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Data Preparation
print("\n[Sonna] Loading local datasets...")
identity_dataset = load_dataset("json", data_files="sonna_identity_mcp.jsonl", split="train")
sft_dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

print("\n[Sonna] Loading HuggingFace SOTA datasets...")
# FineTome-100k for high-quality instruction following
finetome_dataset = load_dataset("mlabonne/FineTome-100k", split="train[:20000]") # Sampling 20k for balance

# OpenThoughts-114k for DeepSeek-R1 style reasoning
reasoning_dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train[:10000]") # Sampling 10k for reasoning

def format_local_prompts(examples):
    prompts = examples.get("prompt", examples.get("instruction", []))
    completions = examples.get("completion", examples.get("output", []))
    texts = []
    for p, c in zip(prompts, completions):
        clean_p = p.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "").strip()
        clean_c = c.replace("<|assistant|>", "").replace("<|end|>", "").strip()
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{clean_p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_c}<|eot_id|>"
        texts.append(text)
    return { "text" : texts, }

def format_finetome(examples):
    conversations = examples["conversations"]
    texts = []
    for conv in conversations:
        # Conv is a list of dicts with 'from' and 'value'
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        for turn in conv:
            role = "user" if turn["from"] == "human" else "assistant"
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{turn['value']}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_reasoning(examples):
    # OpenThoughts typically has 'instruction' and 'thought' + 'output'
    instruction = examples["instruction"]
    thought = examples["thought"]
    output = examples["output"]
    texts = []
    for i, t, o in zip(instruction, thought, output):
        # We wrap thinking in <thought> tags so LM Studio can hide it
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{i}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thought>\n{t}\n</thought>\n\n{o}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

identity_dataset = identity_dataset.map(format_local_prompts, batched=True)
sft_dataset = sft_dataset.map(format_local_prompts, batched=True)
finetome_dataset = finetome_dataset.map(format_finetome, batched=True)
reasoning_dataset = reasoning_dataset.map(format_reasoning, batched=True)

# Combine ALL datasets
# We repeat the identity dataset 10x to make sure it's "unforgettable"
dataset = concatenate_datasets([
    identity_dataset, identity_dataset, identity_dataset, identity_dataset, identity_dataset,
    identity_dataset, identity_dataset, identity_dataset, identity_dataset, identity_dataset,
    sft_dataset,
    finetome_dataset,
    reasoning_dataset
])
dataset = dataset.shuffle(seed=3407)

# 5. Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Packing is better for mixed-length datasets
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 1000, # Increased for larger dataset
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

# 6. Save & Export
print("\n[Sonna] Saving LoRA adapters...")
model.save_pretrained("sonna-4b-lora")
tokenizer.save_pretrained("sonna-4b-lora")

print("\n[Sonna] Exporting to GGUF (Q4_K_M)...")
model.save_pretrained_gguf("sonna-4b-gguf", tokenizer, quantization_method = "q4_k_m")
print("\n[Sonna] DONE.")
