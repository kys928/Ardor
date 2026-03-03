# occipital_gyrus_mapper.py
from pathlib import Path
from tokenizers import Tokenizer
import torch, io, os, random, time, math, glob, tempfile
from tqdm import tqdm

# =========================
# Config (edit as needed)
# =========================
ROOT = Path(__file__).resolve().parent
TOKENIZER = ROOT / ".." / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json"
OUT_DIR   = ROOT / ".." / "Dataset" / "Conversations"  # choose your target folder

CONV_FILES = [
    Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00001.txt"),
    Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\ConversationsOut\Conversations_00002.txt"),
]

# Conversation parsing format
HEAD = "### Conversation"
SEP  = "<eos>"
U_PREFIX = "User: "
A_PREFIX = "Assistant: "

# ---------- safe/atomic saving ----------
def _atomic_torch_save(obj, path, legacy=False):
    d = os.path.dirname(path); os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".part_", suffix=".pt"); os.close(fd)
    try:
        if legacy:
            torch.save(obj, tmp, _use_new_zipfile_serialization=False)
        else:
            torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except OSError: pass
        raise

def _save_shard(buf, out_dir, shard_idx, pad_dtype=torch.int32, legacy=False):
    path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
    obj = {"input_ids": torch.tensor(buf, dtype=pad_dtype)}
    _atomic_torch_save(obj, path, legacy=legacy)

def _save_with_retry(buf, out_dir, shard_idx, pad_dtype=torch.int32):
    try:
        _save_shard(buf, out_dir, shard_idx, pad_dtype, legacy=False)
        return 1
    except Exception as e1:
        print(f"[warn] torch.save failed (zip). Retrying legacy… ({e1})")
        try:
            _save_shard(buf, out_dir, shard_idx, pad_dtype, legacy=True)
            return 1
        except Exception as e2:
            print(f"[warn] legacy save also failed. Splitting… ({e2})")
            n = len(buf)
            if n <= 1:
                raise
            mid = n // 2
            wrote = 0
            wrote += _save_with_retry(buf[:mid], out_dir, shard_idx, pad_dtype)
            wrote += _save_with_retry(buf[mid:], out_dir, shard_idx + wrote, pad_dtype)
            return wrote

def _count_valid_shards(out_dir: str) -> int:
    files = sorted(glob.glob(os.path.join(out_dir, "shard_*.pt")))
    valid = 0
    for f in files:
        try:
            _ = torch.load(f, map_location="cpu")
            valid += 1
        except Exception as e:
            print(f"[warn] removing corrupt shard {f}: {e}")
            try: os.remove(f)
            except OSError: pass
            break
    return valid

# ---------- conversation parsing ----------
def _iter_conversations(path):
    buf = []
    in_conv = False
    with io.open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.rstrip("\n")
            if line == HEAD:
                buf = []; in_conv = True; continue
            if line == SEP and in_conv:
                if buf:
                    yield "\n".join(buf) + "\n<|eot|>"
                buf = []; in_conv = False; continue
            if in_conv:
                if line.startswith(U_PREFIX):
                    buf.append("<|user|> " + line[len(U_PREFIX):])
                elif line.startswith(A_PREFIX):
                    buf.append("<|assistant|> " + line[len(A_PREFIX):])
                elif line:
                    buf.append(line)

# ---------- lightweight utilities for counting & estimating ----------
def _count_total_conversations(paths):
    """Cheap pass: count <eos> that terminate '### Conversation' blocks."""
    total = 0
    in_conv = False
    for p in paths:
        with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line == HEAD:
                    in_conv = True
                elif line == SEP and in_conv:
                    total += 1
                    in_conv = False
    return total

def _sample_avg_tokens_per_conversation(paths, tok, sample_prob=0.001, seed=42):
    """Randomly sample ~0.1% by default to estimate avg tokens/conv."""
    rng = random.Random(seed)
    sampled, token_sum = 0, 0
    for p in paths:
        for conv in _iter_conversations(p):
            if rng.random() <= sample_prob:
                token_sum += len(tok.encode(conv).ids)
                sampled += 1
    if sampled == 0:
        return 0  # fall back later
    return token_sum / sampled

# ---------- main converter with shard cap ----------
def convert_conversations_to_tensors(text_paths,
                                     output_dir,
                                     tokenizer_path,
                                     seq_len=1024,
                                     shard_size=10_000,
                                     sample_prob=0.001,
                                     max_shards=None):
    """
    Convert conversation text files into fixed-length token sequences and save in .pt shards.
    If max_shards is set, total valid shards in output_dir will be capped at that number (resume-safe).
    """
    tok = Tokenizer.from_file(str(tokenizer_path))
    PAD = tok.token_to_id("<pad>")
    assert PAD is not None, "Tokenizer missing <pad>"

    os.makedirs(output_dir, exist_ok=True)

    # ---- RESUME ----
    completed = _count_valid_shards(output_dir)
    shard_idx = completed
    seqs_to_skip = completed * shard_size

    if max_shards is not None and completed >= max_shards:
        print(f"[stop] already have {completed} shards (>= max_shards={max_shards}). Exiting.")
        return

    allowed_new_shards = None if max_shards is None else (max_shards - completed)

    # ---- ESTIMATION ----
    total_convs = _count_total_conversations(text_paths)
    avg_toks = _sample_avg_tokens_per_conversation(text_paths, tok, sample_prob=sample_prob)
    est_total_seqs = None
    if avg_toks > 0:
        est_seqs_per_conv = math.ceil(avg_toks / seq_len)
        est_total_seqs = est_seqs_per_conv * total_convs

    print(f"[resume] found {completed} shards -> skipping first {seqs_to_skip:,} sequences; "
          f"starting at shard_{shard_idx:04d}.pt")
    print(f"[scan] conversations in files ≈ {total_convs:,}")
    if est_total_seqs is not None:
        print(f"[estimate] avg tokens/conv ≈ {avg_toks:.1f} → "
              f"~{est_total_seqs:,} total sequences at seq_len={seq_len}")

    # ---- PROGRESS BAR TOTAL (respect cap) ----
    total_for_bar = None
    if est_total_seqs is not None:
        est_remaining = max(0, est_total_seqs - seqs_to_skip)
        if allowed_new_shards is not None:
            est_remaining = min(est_remaining, allowed_new_shards * shard_size)
        total_for_bar = est_remaining

    # ---- PROCESS ----
    buf = []
    skipped = 0
    seq_emitted = 0
    t0 = time.time()
    last = t0
    done = False

    def shards_written_so_far():
        return shard_idx - completed

    with tqdm(total=total_for_bar, unit="seq") as bar:
        for text_path in text_paths:
            if done: break
            print(f"🔍 Processing {text_path} …")
            for conv_text in _iter_conversations(text_path):
                if done: break
                ids = tok.encode(conv_text).ids
                for start in range(0, len(ids), seq_len):
                    if done: break
                    seq = ids[start:start+seq_len]
                    if len(seq) < seq_len:
                        seq += [PAD] * (seq_len - len(seq))

                    if skipped < seqs_to_skip:
                        skipped += 1
                        if total_for_bar is not None:
                            bar.update(0)
                            bar.set_postfix_str(f"skipped={skipped:,}")
                        continue

                    buf.append(seq)
                    seq_emitted += 1
                    bar.update(1)

                    # ETA/postfix every ~2 seconds
                    now = time.time()
                    if now - last > 2:
                        elapsed = now - t0
                        rate = seq_emitted / max(1e-6, elapsed)
                        if total_for_bar is not None:
                            remaining = max(0, (total_for_bar or 0) - seq_emitted)
                            eta = remaining / max(1e-6, rate)
                            bar.set_postfix(seqs=seq_emitted, rate=f"{rate:.1f}/s", eta=f"{eta/60:.1f}m")
                        else:
                            bar.set_postfix(seqs=seq_emitted, rate=f"{rate:.1f}/s")
                        last = now

                    if len(buf) == shard_size:
                        # Respect capacity BEFORE writing
                        if allowed_new_shards is not None and shards_written_so_far() >= allowed_new_shards:
                            done = True
                            buf.clear()  # drop overflow to keep count exact
                            break

                        prev_written = shards_written_so_far()
                        wrote = _save_with_retry(buf, output_dir, shard_idx, pad_dtype=torch.int32)

                        # Guard against rare overflow (e.g., split save wrote > 1)
                        if allowed_new_shards is not None:
                            remaining = allowed_new_shards - prev_written
                            overflow = max(0, wrote - remaining)
                            if overflow > 0:
                                # Remove extra shards beyond the cap (latest ones)
                                for i in range(overflow):
                                    idx_to_remove = shard_idx + wrote - 1 - i
                                    path = os.path.join(output_dir, f"shard_{idx_to_remove:04d}.pt")
                                    try:
                                        os.remove(path)
                                        print(f"[cap] removed overflow shard {path}")
                                    except OSError as e:
                                        print(f"[warn] couldn't remove overflow shard {path}: {e}")
                                wrote -= overflow
                                done = True  # hit the cap exactly

                        shard_idx += wrote
                        buf.clear()

                        if allowed_new_shards is not None and shards_written_so_far() >= allowed_new_shards:
                            done = True
                            break

            if done: break

    # Final flush (partial shard), only if we still have shard capacity
    if not done and buf:
        if allowed_new_shards is None or (shard_idx - completed) < allowed_new_shards:
            prev_written = shards_written_so_far()
            wrote = _save_with_retry(buf, output_dir, shard_idx, pad_dtype=torch.int32)
            if allowed_new_shards is not None:
                remaining = allowed_new_shards - prev_written
                overflow = max(0, wrote - remaining)
                if overflow > 0:
                    for i in range(overflow):
                        idx_to_remove = shard_idx + wrote - 1 - i
                        path = os.path.join(output_dir, f"shard_{idx_to_remove:04d}.pt")
                        try:
                            os.remove(path)
                            print(f"[cap] removed overflow shard {path}")
                        except OSError as e:
                            print(f"[warn] couldn't remove overflow shard {path}: {e}")
                    wrote -= overflow
            shard_idx += wrote

    print(f"✅ wrote up to shard_{shard_idx-1:04d}.pt in {output_dir}")


