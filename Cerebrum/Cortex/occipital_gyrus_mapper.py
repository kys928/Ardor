# occipital_gyrus_mapper.py
from pathlib import Path
from occipital_input_mapper import convert_conversations_to_tensors

ROOT = Path(__file__).resolve().parent
TOKENIZER = ROOT / ".." / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"
OUT_DIR   = ROOT / ".." / "Dataset" / "Conversations"  # choose your target folder

CONV_FILES = [
    Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00001.txt"),
    Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00002.txt"),
]

convert_conversations_to_tensors(
    text_paths=[str(p) for p in CONV_FILES],
    output_dir=str(OUT_DIR),
    tokenizer_path=str(TOKENIZER),
    seq_len=1024,          # 512/1024/2048 based on model context
    shard_size=40_000,   # sequences per .pt shard
    max_shards=75
)

print("✅ Tensor shards saved:", OUT_DIR)


#first 150 .pt files