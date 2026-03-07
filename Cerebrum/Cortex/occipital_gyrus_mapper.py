# occipital_gyrus_mapper.py
from pathlib import Path
from occipital_input_mapper import convert_conversations_to_tensors

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DATASETS_DIR = REPO_ROOT / "artifacts" / "datasets"

TOKENIZER = REPO_ROOT / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"
OUT_DIR = ARTIFACTS_DATASETS_DIR / "Conversations"

CONV_FILES = [
    DATA_DIR / "Training Data" / "ConversationsOut" / "Conversations_00001.txt",
    DATA_DIR / "Training Data" / "ConversationsOut" / "Conversations_00002.txt",
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
