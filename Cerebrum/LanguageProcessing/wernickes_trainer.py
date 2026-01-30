from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
import os, pathlib


files = [
    "../Training Data/Philosophers/THE REPUBLIC.txt",
    "../Training Data/Philosophers/BEYOND GOOD AND EVIL.txt",
    "../Training Data/Philosophers/Discourse on the Method.txt",
    "../Training Data/Philosophers/Between Heathenism and Christianity.txt",
    "../Training Data/Philosophers/Fear and Trembling.txt",
    "../Training Data/Philosophers/Modern Man in Search of A Soul.txt",
    "../Training Data/Philosophers/Myth of Sisyphus.txt",
    "../Training Data/Philosophers/THE INTERPRETATION OF DREAMS.txt",
    "../Training Data/Philosophers/The Prince.txt",
    "../Training Data/Philosophers/The Metamorphosis.txt",
    "../Training Data/Philosophers/BEING AND NOTHINGNESS.txt",
    "../Training Data/Philosophers/The Brothers Karamazov.txt",
    "../Training Data/Philosophers/AN ENQUIRY CONCERNING HUMAN UNDERSTANDING.txt",
    "../Training Data/Philosophers/An Essay Concerning Human Understanding.txt",
    "../Training Data/Philosophers/NOVUM ORGANUM.txt",

    "../Training Data/Scientists/A Brief History of Time.txt",
    "../Training Data/Scientists/COMPUTING MACHINERY.txt",
    "../Training Data/Scientists/Dialogue.txt",
    "../Training Data/Scientists/On the Origin of Species.txt",
    "../Training Data/Scientists/Physics and Philosophy.txt",
    "../Training Data/Scientists/PRINCIPIA.txt",
    "../Training Data/Scientists/Relativity.txt",
    "../Training Data/Scientists/Researches.txt",
    "../Training Data/Scientists/WHAT IS LIFE.txt",

    "../Training Data/Leaders/Commentaries.txt",
    "../Training Data/Leaders/CORRESPONDENCE.txt",
    "../Training Data/Leaders/MARXISM AND THE NATIONAL QUESTION.txt",
    "../Training Data/Leaders/Meditations.txt",
    "../Training Data/Leaders/MEIN KAMPF.txt",
    "../Training Data/Leaders/STATE AND REVOLUTION.txt",
    "../Training Data/Leaders/THE ART OF WAR.txt",
    "../Training Data/Leaders/The Book of Five Rings.txt",
    "../Training Data/Leaders/The story of my experiments with truth.txt",

]
all_texts = [p for p in files if pathlib.Path(p).is_file()]
assert all_texts, "❌ No training files found!"

# ------------------------------------------------------------------
# 2. tokenizer training exactly as before
output_path = "../ProjectTokenizer/ardor_tokenizer/tokenizer_v5.json"

tok = Tokenizer(models.BPE())
tok.normalizer     = normalizers.Sequence([normalizers.NFC()])
tok.pre_tokenizer  = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=52_224,
    min_frequency=2,
    special_tokens=["<pad>", "<eos>"],
    show_progress=True
)

print(f"📚 Training on {len(all_texts)} files")
tok.train(files=all_texts, trainer=trainer)

tok.post_processor = processors.ByteLevel(trim_offsets=True)

tok.save(output_path)
print(f"✅ Saved tokenizer with vocab size {len(tok.get_vocab())} → {output_path}")
