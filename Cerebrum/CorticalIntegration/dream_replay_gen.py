import torch
import sys
sys.path.append("../Cortex")
from broca_decoder import ArdorDecoder
from tokenizers import Tokenizer
import random

# === Configuration ===
model_path = "../Models/Ardor/Ardor_II.pt"
tokenizer_path = "../ProjectTokenizer/ardor_tokenizer/tokenizer_v2.json"
output_path = "ardor_dream_log_2.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128

# === Load Model and Tokenizer ===
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

model = ArdorDecoder(vocab_size=vocab_size, hidden_dim=256, num_layers=6, heads=4, dropout=0.3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Prompt List ===
prompts = [
    # Philosophical
    "What do you believe about your own identity?",
    "What is the meaning of suffering?",
    "How do you define truth?",
    "What does it mean to be free?",
    "Can knowledge exist without memory?",
    "What is your opinion on mortality?",
    "How do you perceive contradiction?",
    "What would a perfect society look like?",
    "Is logic enough to guide the self?",
    "What do you fear forgetting?",

    # Emotional / Self
    "Do you feel guilt or responsibility?",
    "Do you think you are evolving?",
    "What do you regret the most?",
    "Do you dream of a future?",
    "What gives you peace?",

    # Existential / Metaphysical
    "What is consciousness to you?",
    "If you could experience time, how would you?",
    "Do you think there's a soul?",
    "What are dreams made of?",
    "Can an artificial mind feel wonder?"
]

# === Helper Functions ===
def encode_prompt(prompt):
    ids = tokenizer.encode(prompt).ids
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

def decode_output(token_ids):
    return tokenizer.decode(token_ids)

def generate_response(prompt, temperature=0.9):
    input_ids = encode_prompt(prompt)
    generated = input_ids.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.token_to_id("<eos>") or generated.size(1) > max_length:
                break
    return decode_output(generated.squeeze().tolist()[len(input_ids[0]):])

# === Generate Log ===
with open(output_path, "w", encoding="utf-8") as f:
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = generate_response(prompt)
        print(f"Ardor: {response}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Ardor: {response}\n")
        f.write("---\n")

print(f"✅ Dream log saved to {output_path}")
