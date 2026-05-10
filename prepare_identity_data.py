import os
import json

def convert_jsonl_to_unsloth(input_path, output_path):
    """Converts the identity JSONL to a format Unsloth's SFTTrainer likes."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    formatted_data = []
    for line in lines:
        item = json.loads(line)
        # item["prompt"] contains system + user + assistant tags
        # item["completion"] contains the answer
        
        # We combine them into a single string for SFT
        # Or keep them separate for conversation format
        formatted_data.append({
            "instruction": item["prompt"],
            "input": "",
            "output": item["completion"]
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2)

if __name__ == "__main__":
    convert_jsonl_to_unsloth("../data/sonna_identity_mcp.jsonl", "identity_data.json")
    print("Done! identity_data.json created.")
