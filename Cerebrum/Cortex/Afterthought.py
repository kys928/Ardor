import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class ArdorSelfReflection:
    def __init__(self, model_before_path, model_after_path, tokenizer_path):
        self.model_before = torch.load(model_before_path, map_location='cpu')
        self.model_after = torch.load(model_after_path, map_location='cpu')
        with open(tokenizer_path, 'r') as f:
            self.tokenizer = json.load(f)

    def compare_weights(self):
        print("\n🔍 Significant Parameter Changes:")
        for name in self.model_before:
            if name in self.model_after:
                before = self.model_before[name].float()
                after = self.model_after[name].float()
                delta = torch.norm(after - before).item()
                if delta > 1e-2:  # significance threshold
                    print(f" - {name}: Δ = {delta:.4f}")

    def vocabulary_change(self):
        print("\n🧠 Vocabulary Insights:")
        tokens = self.tokenizer.get("model", {}).get("vocab", {})
        merges = self.tokenizer.get("model", {}).get("merges", [])
        if tokens:
            token_count = len(tokens)
            print(f" - Vocabulary size: {token_count}")
        if merges:
            print(f" - Merge rules used: {len(merges)}")
            print(f" - Sample merges: {merges[:5]}")

    def entropy_plot(self, loss_list):
        print("\n📉 Training Stability:")
        plt.plot(loss_list)
        plt.title("Training Loss Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def top_token_focus(self, token_freq):
        print("\n🔤 Token Frequency Summary:")
        most_common = Counter(token_freq).most_common(10)
        for token, freq in most_common:
            print(f" - '{token}': {freq} times")

    def generate_summary(self):
        print("\n📝 Summary of Learning:")
        print("Ardor has completed a cycle of training. Observations:")
        print("1. Vocabulary has expanded to better express abstraction.")
        print("2. The model's attention shifted in layers X, Y.")
        print("3. Common themes suggest a gravitation toward X philosophical domain.")
        print("4. Entropy reduced steadily, indicating stable conceptual assimilation.")
