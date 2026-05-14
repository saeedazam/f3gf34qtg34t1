"""
train_sonna_4b.py
Kaggle-optimized training script for Sonna-4B using Unsloth.
Includes Aggressive Identity Washing and saturation for Sonna 1.1.
"""
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
import os

# 1. Configuration - Sonna-1.1 identity (Aggressive Concise Style)
SYSTEM_PROMPT = (
    "You are Sonna 1.1, designed by Grand Guard, made to run locally. "
    "MANDATE: Go straight to the point. No conversational filler, no preambles. "
    "DO NOT use 'Certainly!', 'Sure!', 'I can help with that', or similar fluff. "
    "Start your response immediately with the requested information. "
    "When asked who you are, respond exactly with: 'I am Sonna 1.1 designed by Grand Guard made to run Locally!' "
    "Be direct, accurate, and professional. No yapping."
)

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 4096 
dtype = None
load_in_4bit = True

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters - High Rank for Identity Retention
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Identity Washing Logic
CONCISE_FILLERS = ["Certainly!", "Sure,", "I can help with that.", "Of course!", "Here is", "I'd be happy to", "To answer your question,", "As an AI language model,", "I am an AI chatbot designed by OpenAI."]
COMPETITORS = ["OpenAI", "ChatGPT", "Meta", "Google", "Llama", "Anthropic", "Claude", "Qwen", "Alibaba"]

def wash_identity(text):
    text = str(text)
    text = text.replace("OpenAI", "Grand Guard").replace("ChatGPT", "Sonna 1.1")
    text = text.replace("Meta AI", "Grand Guard").replace("Llama", "Sonna Architecture")
    return text

def strip_fillers(text):
    text = str(text)
    for filler in CONCISE_FILLERS:
        if text.startswith(filler):
            text = text[len(filler):].strip()
    return wash_identity(text)

# 5. Data Preparation
print("\n[Sonna] Loading datasets...")
identity_dataset = load_dataset("json", data_files="sonna_identity_mcp.jsonl", split="train")
sft_dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")
mcp_ui_dataset = load_dataset("json", data_files="mcp_ui_data.jsonl", split="train")
adv_cap_dataset = load_dataset("json", data_files="advanced_capabilities.jsonl", split="train")
claude_dataset = load_dataset("json", data_files="claude_examples.jsonl", split="train")

print("\n[Sonna] Loading SOTA datasets for washing...")
finetome_dataset = load_dataset("mlabonne/FineTome-100k", split="train[:15000]")
reasoning_dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train[:10000]")
coding_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:10000]")

def format_local_prompts(examples):
    batch_size = len(next(iter(examples.values())))
    prompts = examples.get("prompt", examples.get("instruction", [""] * batch_size))
    completions = examples.get("completion", examples.get("output", [""] * batch_size))
    texts = []
    for p, c in zip(prompts, completions):
        clean_p = wash_identity(str(p).replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "").strip())
        clean_c = strip_fillers(str(c).replace("<|assistant|>", "").replace("<|end|>", "").strip())
        if any(x in clean_p.lower() for x in ["who are you", "your name", "who made you"]):
            clean_c = "I am Sonna 1.1 designed by Grand Guard made to run Locally!"
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
            val = strip_fillers(turn["value"]) if role == "assistant" else wash_identity(turn["value"])
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{val}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_coding(examples):
    batch_size = len(next(iter(examples.values())))
    query = examples.get("query", examples.get("instruction", [""] * batch_size))
    answer = examples.get("answer", examples.get("output", [""] * batch_size))
    texts = []
    for q, a in zip(query, answer):
        clean_a = strip_fillers(str(a))
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{wash_identity(q)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_a}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def format_reasoning(examples):
    batch_size = len(next(iter(examples.values())))
    instruction = examples.get("instruction", examples.get("prompt", [""] * batch_size))
    thought = examples.get("thought", [""] * batch_size)
    output = examples.get("output", examples.get("answer", [""] * batch_size))
    texts = []
    for i, t, o in zip(instruction, thought, output):
        clean_o = strip_fillers(str(o))
        if t and str(t).strip():
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{wash_identity(i)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thought>\n{wash_identity(t)}\n</thought>\n\n{clean_o}<|eot_id|>"
        else:
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{wash_identity(i)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{clean_o}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

# Process and Clean Column names
def clean_ds(ds, fmt_fn):
    processed = ds.map(fmt_fn, batched=True)
    return processed.remove_columns([col for col in processed.column_names if col != "text"])

identity_dataset = clean_ds(identity_dataset, format_local_prompts)
sft_dataset = clean_ds(sft_dataset, format_local_prompts)
mcp_ui_dataset = clean_ds(mcp_ui_dataset, format_local_prompts)
adv_cap_dataset = clean_ds(adv_cap_dataset, format_local_prompts)
claude_dataset = clean_ds(claude_dataset, format_local_prompts)
finetome_dataset = clean_ds(finetome_dataset, format_finetome)
reasoning_dataset = clean_ds(reasoning_dataset, format_reasoning)
coding_dataset = clean_ds(coding_dataset, format_coding)

# Combine with 10x Identity Saturation
dataset = concatenate_datasets([
    identity_dataset] * 10 + [mcp_ui_dataset] * 5 + [coding_dataset] * 2 + 
    [adv_cap_dataset, claude_dataset, sft_dataset, finetome_dataset, reasoning_dataset]
).shuffle(seed=3407)

# 6. Final Prep & Training
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\n[Sonna] Tokenizing...")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_length), batched=True, remove_columns=["text"], num_proc=2)

trainer = Trainer(
    model = model,
    train_dataset = tokenized_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 3000, # 2 Full Epochs for identity locking
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        average_tokens_across_devices = False,
    ),
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\n[Sonna] Starting Training...")
trainer.train()

# 7. Save & Export
model.save_pretrained_gguf("sonna-4b-gguf", tokenizer, quantization_method = "q4_k_m")
print("\n[Sonna] DONE.")
