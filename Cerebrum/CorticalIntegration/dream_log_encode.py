import os
import torch
from tokenizers import Tokenizer

# === Configuration ===
tokenizer_path = "../ProjectTokenizer/ardor_tokenizer/tokenizer_v3.json"  # or v2 if you're using Ardor_II

output_tensor_path = "ardor_dream.pt"
dream_folder = "../CorticalIntegration/Dreams"
unified_log_path = "ardors_dreams.txt"
block_size = 128

# === Load Tokenizer ===
tokenizer = Tokenizer.from_file(tokenizer_path)

# === Helper: Merge all dream logs into one file ===
def merge_dream_logs(folder_path, output_path):
    with open(output_path, "w", encoding="utf-8") as out_file:
        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                    out_file.write(f.read())
                    out_file.write("\\n")

# === Helper: Tokenize and encode log lines ===
def encode_dream_log(log_path):
    sequences = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("---") or line.lower().startswith("prompt:"):
            continue
        ids = tokenizer.encode(line).ids
        if len(ids) >= 2:
            ids = ids[:block_size]  # Truncate long ones
            sequences.append(torch.tensor(ids, dtype=torch.long))

    return sequences

# === Run merge
merge_dream_logs(dream_folder, unified_log_path)
print(f"✅ Unified dreams written to {unified_log_path}")

# === Encode main dream log
sequences = encode_dream_log(unified_log_path)
print(f"📜 Encoded {len(sequences)} dream lines.")

# === Pad and stack sequences
max_len = max(len(seq) for seq in sequences)
padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = seq

# === Save tensor
torch.save(padded, output_tensor_path)
print(f"💾 Saved dream tensor to {output_tensor_path}")
