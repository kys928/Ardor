# wernickes_area.py (tokenizer builder)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
SAVE_PATH = (ROOT / ".." / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json").resolve()

# Two conversation shards (fix the typo in your path)
texts = [
    r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00001.txt",
    r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00002.txt",
]

# ── Build Byte-Level BPE Tokenizer ─────────────────────────────────────
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# Keep punctuation normalization; avoid lowercasing/strip-accents to preserve dialog style
tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace("“", '"'), normalizers.Replace("”", '"'),
    normalizers.Replace("‘", "'"), normalizers.Replace("’", "'"),
    normalizers.Replace("—", "-"), normalizers.Replace("–", "-"),
    normalizers.Replace("…", "..."),
    normalizers.NFKC(),
])

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder       = decoders.ByteLevel()

special_tokens = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|eot|>"
]

trainer = trainers.BpeTrainer(
    vocab_size=52_224,
    min_frequency=2,
    special_tokens=special_tokens,
    show_progress=True
)

tokenizer.train(files=texts, trainer=trainer)

# Add BOS/EOS automatically on encode
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> <bos> $B:1 <eos>:1",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

os.makedirs(SAVE_PATH.parent, exist_ok=True)
tokenizer.save(str(SAVE_PATH))
print(f"✅ Tokenizer saved to {SAVE_PATH}")
