# Sonna 1.1 Kaggle Training Guide

This folder contains everything you need to train **Sonna 1.1** on Kaggle.

### How to use on Kaggle:
1.  **Upload:** Create a new Dataset on Kaggle and upload all files in this directory.
2.  **Notebook:** Create a new Notebook and add the dataset you just created.
3.  **Accelerator:** Set the Accelerator to **GPU T4 x2** or **GPU P100**.
4.  **Internet:** Ensure **Internet on** is toggled in the settings (needed for Unsloth and HF datasets).
5.  **Run:** Open `Sonna_Training.ipynb` (or copy-paste its content) and run all cells.

### Features Included:
- **Thinking Box:** Model will use `<thought>...</thought>` tags before answering.
- **MCP Training:** Specialized data for tool calling and Model Context Protocol.
- **UI Optimization:** Markdown headings, emojis, and clean formatting.
- **Identity:** Fixed identity: "I am Sonna 1.1 designed by Grand Guard made to run Locally!"

### File Structure:
- `train_sonna_4b.py`: Main training logic using Unsloth.
- `Sonna_Training.ipynb`: Kaggle-ready execution environment.
- `*.jsonl`: Consolidated SFT datasets (Identity, MCP, UI, Reasoning).
