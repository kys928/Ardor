# decode_gravis_shard.py
# import torch
# from tokenizers import Tokenizer
#
# # --- Full explicit paths ---
# SHARD_PATH = r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Dataset\GravisCorpora_shards\shard_0000.pt"
# TOKENIZER_PATH = r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\ProjectTokenizer\ardor_tokenizer\tokenizer_v6.json"
# OUTPUT_TXT = r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Training Data\GravisCorpora_shard0.txt"
#
# # Load tokenizer and shard
# tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
# shard = torch.load(SHARD_PATH, map_location="cpu")
#
# # Decode
# print(f"🔍 Decoding {SHARD_PATH} → {OUTPUT_TXT}")
# with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
#     for seq in shard:
#         text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
#         f.write(text.strip() + "\n")
#
# print(f"✅ Saved decoded text to {OUTPUT_TXT}")


# from PIL import Image
# from pathlib import Path
# root = Path(r"C:\Users\adm\PycharmProjects\ProjectArdor1.2")
# src = "C:/Users/adm/PycharmProjects/ProjectArdor/Praetor/ArdorApp/assets/ArdorImg.webp"
# dst = "C:/Users/adm/PycharmProjects/ProjectArdor/Praetor/ArdorApp/assets/ardor.ico"
#
# img = Image.open(src).convert("RGBA")
# # multi-size ICO for crisp small/large tiles
# sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]
# img.save(dst, format="ICO", sizes=sizes)
# print("Wrote", dst)


# harvest_conversations.py
# import os, json, re, hashlib, itertools
# from pathlib import Path
# from datasets import load_dataset
# from datetime import datetime
#
# TARGET_DIALOGS = int(os.environ.get("TARGET_DIALOGS", "10000000"))  # 10M, change if needed
# MAX_TURNS_PER_DIALOG = 16
# MIN_TURN_CHARS, MAX_TURN_CHARS = 3, 2000
# ROTATE_BYTES = 2_000_000_000  # rotate ~2GB per file
# OUT_DIR = Path(os.environ.get("OUT_DIR", "./ConversationsOut"))
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# def clean(s: str) -> str:
#     if not s: return ""
#     s = s.replace("\r", "").strip()
#     s = re.sub(r"\s+\n", "\n", s)
#     s = re.sub(r"\n{3,}", "\n\n", s)
#     return s
#
# def valid_turn(s: str) -> bool:
#     if not s: return False
#     s = s.strip()
#     return MIN_TURN_CHARS <= len(s) <= MAX_TURN_CHARS
#
# def write_dialog(fh, messages):
#     # messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
#     # flatten to alternating User/Assistant (system lines prefixed explicitly)
#     lines = ["### Conversation"]
#     for m in messages:
#         role = m["role"]
#         if role == "user":
#             lines.append("User: " + m["content"])
#         elif role == "assistant":
#             lines.append("Assistant: " + m["content"])
#         else:
#             lines.append("System: " + m["content"])
#     lines.append("<eos>")
#     fh.write("\n".join(lines) + "\n")
#
# def rotator(base_dir: Path):
#     idx, bytes_written = 1, 0
#     fh = open(base_dir / f"Conversations_{idx:05d}.txt", "w", encoding="utf-8")
#     while True:
#         line = yield fh  # get called to check rotation
#         if line is None:
#             # on close
#             fh.close()
#             return
#         size = len(line.encode("utf-8")) + 1
#         if bytes_written + size > ROTATE_BYTES:
#             fh.close()
#             idx += 1
#             bytes_written = 0
#             fh = open(base_dir / f"Conversations_{idx:05d}.txt", "w", encoding="utf-8")
#         fh.write(line + "\n")
#         bytes_written += size
#
# # ---------------- Loaders per source ----------------
#
# def role_norm(r, turn_idx):
#     r = (r or "").lower()
#     if r in ("user","human","prompter","client"): return "user"
#     if r in ("assistant","gpt","bot","model"):    return "assistant"
#     if r in ("system",):                           return "system"
#     return "user" if (turn_idx % 2 == 0) else "assistant"
#
# def from_lmsys():
#     # gated: accept terms on HF
#     ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
#     for ex in ds:
#         raw = ex.get("conversation")
#         try:
#             msgs = json.loads(raw) if isinstance(raw, str) else raw
#         except Exception:
#             continue
#         out=[]
#         for i, m in enumerate(msgs):
#             role = role_norm(m.get("role"), i)
#             content = clean(m.get("content"))
#             if valid_turn(content):
#                 out.append({"role": role, "content": content})
#             if len(out) >= MAX_TURNS_PER_DIALOG: break
#         if sum(1 for m in out if m["role"]=="user") and sum(1 for m in out if m["role"]=="assistant"):
#             yield out
#
# def from_dialogstudio():
#     # gated: accept terms on HF; choose a few big ones
#     for subset in ("DAILYDIALOG","MULTIWOZ2_2","TASKMASTER1","TASKMASTER2"):
#         try:
#             d = load_dataset("Salesforce/dialogstudio", name=subset)
#         except Exception:
#             continue
#         split = d.get("train") or next(iter(d.values()))
#         for ex in split:
#             turns = ex.get("dialogue") or ex.get("conversation") or ex.get("turns")
#             if not turns: continue
#             out=[]
#             for i, t in enumerate(turns):
#                 role = role_norm(t.get("role") or t.get("speaker"), i)
#                 content = clean(t.get("text") or t.get("utterance"))
#                 if valid_turn(content):
#                     out.append({"role": role, "content": content})
#                 if len(out) >= MAX_TURNS_PER_DIALOG: break
#             if sum(1 for m in out if m["role"]=="user") and sum(1 for m in out if m["role"]=="assistant"):
#                 yield out
#
# def from_ultrachat():
#     # open; ~200k multi-turn dialogs (filtered). :contentReference[oaicite:1]{index=1}
#     d = load_dataset("HuggingFaceH4/ultrachat_200k")
#     for split_name in d:
#         split = d[split_name]
#         for ex in split:
#             # common fields: 'messages' or 'prompt'/'response' lists depending on subset version
#             msgs = ex.get("messages")
#             out=[]
#             if msgs:
#                 for i, m in enumerate(msgs):
#                     role = role_norm(m.get("role") or m.get("from"), i)
#                     content = clean(m.get("content") or m.get("value"))
#                     if valid_turn(content):
#                         out.append({"role": role, "content": content})
#                     if len(out) >= MAX_TURNS_PER_DIALOG: break
#             else:
#                 # fallback 2-turn
#                 p = clean(ex.get("prompt") or "")
#                 r = clean(ex.get("response") or "")
#                 if valid_turn(p) and valid_turn(r):
#                     out = [{"role":"user","content":p},{"role":"assistant","content":r}]
#             if len(out) >= 2:
#                 yield out
#
# def from_oasst1():
#     # open; conversation trees; we linearize root->leaf English paths. :contentReference[oaicite:2]{index=2}
#     d = load_dataset("OpenAssistant/oasst1")
#     for split_name in d:
#         split = d[split_name]
#         for ex in split:
#             # parquet rows are messages; group by 'conversation_id' if present
#             # but the HF card warns this is not perfectly "nested"; simplest: try prompt->reply pairs
#             lang = (ex.get("lang") or ex.get("language") or "en").lower()
#             if lang != "en": continue
#             role = (ex.get("role") or ex.get("message_role") or "").lower()
#             text = clean(ex.get("text") or ex.get("message") or "")
#             if not valid_turn(text): continue
#             # We only build simple 2-turns when a reply_text is available on the row
#             reply = clean(ex.get("reply") or ex.get("assistant_response") or "")
#             if valid_turn(reply):
#                 yield [{"role":"user","content":text},{"role":"assistant","content":reply}]
#
# def from_gutenberg_pairs():
#     # HF mirror of Gutenberg dialogue utterances; pair consecutive lines. Size 10M–100M utterances. :contentReference[oaicite:3]{index=3}
#     for split in ("train","dev","test"):
#         try:
#             ds = load_dataset("willwade/Gutenberg-dialog-en", split=split, streaming=True)
#         except Exception:
#             continue
#         buf=[]
#         for ex in ds:
#             line = clean(ex.get("text") or "")
#             if not valid_turn(line):
#                 continue
#             buf.append(line)
#             if len(buf) >= 2:
#                 u = buf.pop(0); a = buf.pop(0) if buf else None
#                 if a and valid_turn(a):
#                     yield [{"role":"user","content":u},{"role":"assistant","content":a}]
#
# # --------------- Main ---------------
#
# def harvest():
#     sources = [
#         ("lmsys", from_lmsys),
#         ("dialogstudio", from_dialogstudio),
#         ("ultrachat", from_ultrachat),
#         ("oasst1", from_oasst1),
#         ("gutenberg", from_gutenberg_pairs),
#     ]
#     count = 0
#     shard_idx, bytes_written = 1, 0
#     fh = open(OUT_DIR / f"Conversations_{shard_idx:05d}.txt", "w", encoding="utf-8")
#
#     def rotate_if_needed(extra_bytes):
#         nonlocal fh, shard_idx, bytes_written
#         if bytes_written + extra_bytes > ROTATE_BYTES:
#             fh.close()
#             shard_idx += 1
#             bytes_written = 0
#             fh = open(OUT_DIR / f"Conversations_{shard_idx:05d}.txt", "w", encoding="utf-8")
#
#     for name, gen in sources:
#         try:
#             print(f"[{datetime.now().strftime('%H:%M:%S')}] -> Harvesting {name} ...")
#             for messages in gen():
#                 if len(messages) < 2:
#                     continue
#                 # render once to measure
#                 rendered = ["### Conversation"]
#                 for m in messages:
#                     r = "User:" if m["role"]=="user" else ("Assistant:" if m["role"]=="assistant" else "System:")
#                     rendered.append(f"{r} {m['content']}")
#                 rendered.append("<eos>")
#                 rendered.append("")  # spacer
#                 blob = "\n".join(rendered) + "\n"
#
#                 rotate_if_needed(len(blob.encode("utf-8")))
#                 fh.write(blob)
#                 bytes_written += len(blob.encode("utf-8"))
#                 count += 1
#                 if count % 10000 == 0:
#                     print(f"  ... {count:,} dialogs written")
#                 if count >= TARGET_DIALOGS:
#                     fh.close()
#                     print(f"[DONE] Wrote {count:,} dialogs across {shard_idx} shard(s) → {OUT_DIR}")
#                     return
#         except Exception as e:
#             print(f"[SKIP] {name}: {e}")
#
#     fh.close()
#     print(f"[END] Sources exhausted. Wrote {count:,} dialogs across {shard_idx} shard(s) → {OUT_DIR}")
#
# if __name__ == "__main__":
#     harvest()


# scan_pt_integrity.py
# List corrupted/ok/unknown .pt shards without moving anything.
# Usage:
#   python scan_pt_integrity.py --dir "Dataset/Conversations" --pattern "shard_*.pt" --save-json bad_shards.json
# Flags:
#   --fast        : skip deep torch.load() on non-ZIP files (faster, may mark some as 'unknown')
#   --workers 8   : parallelize ZIP checks (I/O bound)
#   --verbose     : print per-file status

# import argparse, json, csv, zipfile, sys
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# import torch
#
# EXPECTED_ZIP_MEMBERS = {
#     "data.pkl", "archive/data.pkl", "pytorch_metadata.json", "version"
# }
#
# def is_zip_ok(p: Path) -> (bool, str):
#     try:
#         with zipfile.ZipFile(p, "r") as zf:
#             # Check internal consistency
#             bad = zf.testzip()
#             if bad is not None:
#                 return False, f"zip member CRC fail: {bad}"
#             names = set(zf.namelist())
#             if not (names & EXPECTED_ZIP_MEMBERS):
#                 # Not strictly required, but helpful sanity check
#                 return False, "zip missing expected torch members"
#         return True, ""
#     except Exception as e:
#         return False, f"zip open/test error: {e}"
#
# def deep_load_ok(p: Path) -> (bool, str, str):
#     """Attempt a full torch.load on CPU. Returns (ok, type, reason)."""
#     try:
#         obj = torch.load(p, map_location="cpu")
#         # Optional: validate shard shape if you expect [N, T] LongTensor
#         if isinstance(obj, torch.Tensor):
#             if obj.ndim == 2 and obj.dtype in (torch.long, torch.int64):
#                 return True, "tensor2d", ""
#             else:
#                 return True, f"tensor{obj.ndim}", "non-2D or non-int64 (may still be valid)"
#         elif isinstance(obj, dict):
#             return True, "state_dict", ""
#         else:
#             return True, type(obj).__name__, ""
#     except MemoryError:
#         return False, "unknown", "MemoryError during torch.load"
#     except Exception as e:
#         return False, "legacy", f"torch.load error: {e}"
#
# def quick_header(p: Path) -> bytes:
#     try:
#         with open(p, "rb") as f:
#             return f.read(4)
#     except Exception:
#         return b""
#
# def check_one(p: Path, fast: bool, verbose: bool) -> dict:
#     header = quick_header(p)
#     # Flag obvious temp/partial files by name
#     if p.name.startswith(".part__"):
#         res = {"file": str(p), "status": "corrupt", "kind": "partial", "reason": "partial/temp filename"}
#         if verbose: print("CORRUPT  ", p.name, "→ partial temp")
#         return res
#
#     if header == b'PK\x03\x04':  # ZIP-based torch save
#         ok, reason = is_zip_ok(p)
#         status = "ok" if ok else "corrupt"
#         kind = "zip"
#         if verbose:
#             print(f"{status.upper():8} {p.name}  (zip) {'' if ok else reason}")
#         return {"file": str(p), "status": status, "kind": kind, "reason": reason}
#     else:
#         # Legacy or non-zip; optionally do deep load
#         if fast:
#             if verbose: print("UNKNOWN  ", p.name, "(legacy/non-zip; use --fast off for deep check)")
#             return {"file": str(p), "status": "unknown", "kind": "legacy", "reason": "fast mode; not loaded"}
#         ok, kind, reason = deep_load_ok(p)
#         status = "ok" if ok else "corrupt"
#         if verbose:
#             print(f"{status.upper():8} {p.name}  ({kind}) {'' if ok else reason}")
#         return {"file": str(p), "status": status, "kind": kind, "reason": reason}
#
# def main():
#     ap = argparse.ArgumentParser(description="Scan .pt shards and list corrupted files.")
#     ap.add_argument("--dir", default="Dataset/Conversations", help="Root directory to scan")
#     ap.add_argument("--pattern", default="*.pt", help="Glob pattern, e.g. shard_*.pt")
#     ap.add_argument("--fast", action="store_true", help="Skip deep torch.load for legacy files")
#     ap.add_argument("--workers", type=int, default=8, help="Threads for ZIP checks")
#     ap.add_argument("--save-json", default=None, help="Write full report to JSON")
#     ap.add_argument("--save-csv", default=None, help="Write full report to CSV")
#     ap.add_argument("--verbose", action="store_true", help="Print per-file status")
#     args = ap.parse_args()
#
#     root = Path(args.dir)
#     files = sorted(root.rglob(args.pattern))
#     if not files:
#         print(f"No files match {args.pattern} under {root.resolve()}")
#         sys.exit(0)
#
#     # Strategy: parallelize checks; deep loads are done in threads too, but you can set workers=1 if you prefer
#     results = []
#     with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
#         futs = {ex.submit(check_one, p, args.fast, args.verbose): p for p in files}
#         for fut in as_completed(futs):
#             results.append(fut.result())
#
#     # Summaries
#     ok = [r for r in results if r["status"] == "ok"]
#     corrupt = [r for r in results if r["status"] == "corrupt"]
#     unknown = [r for r in results if r["status"] == "unknown"]
#
#     print("\n=== Scan Summary ===")
#     print(f"Total   : {len(results)}")
#     print(f"OK      : {len(ok)}")
#     print(f"CORRUPT : {len(corrupt)}")
#     print(f"UNKNOWN : {len(unknown)} (run without --fast for deeper check)")
#
#     if corrupt:
#         print("\nCorrupted files:")
#         for r in corrupt[:200]:  # cap display; still saved in reports
#             print(" -", r["file"], "→", r.get("reason",""))
#
#     # Save reports
#     if args.save_json:
#         Path(args.save_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
#         print(f"\nJSON report written to {args.save_json}")
#     if args.save_csv:
#         with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=["file","status","kind","reason"])
#             w.writeheader()
#             w.writerows(results)
#         print(f"CSV report written to {args.save_csv}")
#
# if __name__ == "__main__":
#     main()

# import torch
# import torch.nn.functional as F
# from tokenizers import Tokenizer
# from pathlib import Path
# import sys
# sys.path.append("../Cerebrum/Cortex")
# from broca_decoder import ArdorDecoder
#
# # === Paths ===
# MODEL_PATH = Path("../Cerebrum/Models/Ardor/Ardor_Final.pt")
# TOKENIZER_PATH = Path("../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json")
#
# # === Load tokenizer ===
# tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
# bos_id = tokenizer.token_to_id("<bos>")
# eos_id = tokenizer.token_to_id("<eos>")
#
# # === Load model ===
# model = ArdorDecoder(vocab_size=52224, hidden=384, layers=8, heads=6)
# model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
# model.eval()
#
# # === Generation function ===
# def generate(prompt, max_new_tokens=50, temperature=0.9, top_k=40, stream=True):
#     # Encode input
#     encoded = tokenizer.encode(prompt)
#     input_ids = torch.tensor([[bos_id] + encoded.ids], dtype=torch.long)
#
#     if stream:
#         print("\n=== Ardor Generation (live) ===")
#         print("Prompt:", prompt)
#         print("Response: ", end="", flush=True)
#
#     for _ in range(max_new_tokens):
#         with torch.no_grad():
#             logits = model(input_ids)
#
#         # Focus on last token
#         logits = logits[:, -1, :] / temperature
#
#         # Top-k filter
#         if top_k is not None:
#             vals, idxs = torch.topk(logits, top_k)
#             probs = F.softmax(vals, dim=-1)
#             next_id = idxs[0, torch.multinomial(probs, 1)]
#         else:
#             probs = F.softmax(logits, dim=-1)
#             next_id = torch.multinomial(probs, 1)
#
#         # Fix shape mismatch (make sure it's [1,1])
#         next_id = next_id.view(1, 1)
#
#         # Append
#         input_ids = torch.cat([input_ids, next_id], dim=1)
#
#         # Decode only new token
#         token_text = tokenizer.decode([next_id.item()])
#
#         if stream:
#             print(token_text, end="", flush=True)
#
#         # Stop if EOS
#         if next_id.item() == eos_id:
#             break
#
#     if stream:
#         print("\n==============================\n")
#         return None
#     else:
#         return tokenizer.decode(input_ids[0].tolist())
#
#
# # === Test prompt ===
# if __name__ == "__main__":
#     prompt = "What would you like to do?"
#     _ = generate(prompt, max_new_tokens=60, temperature=0.8, top_k=40, stream=True)









# diagnostics_artifact_probe.py
# import torch
# import torch.nn.functional as F
# from tokenizers import Tokenizer
# from pathlib import Path
# import numpy as np
# import sys
# sys.path.append("../Cerebrum/Cortex")
# from broca_decoder import ArdorDecoder
#
# # === Paths ===
# MODEL_PATH = Path("../Cerebrum/Models/Ardor/Ardor_Ksai.pt")   # or Final, Nova, Ksai
# TOK_V7 = Path("../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v7.json")
# TOK_V8 = Path("../Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json")
#
# # === Load tokenizers ===
# tok7 = Tokenizer.from_file(str(TOK_V7))
# tok8 = Tokenizer.from_file(str(TOK_V8))
#
# vocab7 = {t: i for t, i in tok7.get_vocab().items()}
# vocab8 = {t: i for t, i in tok8.get_vocab().items()}
#
# shared_tokens = sorted(set(vocab7.keys()) & set(vocab8.keys()))
# unique7 = sorted(set(vocab7.keys()) - set(vocab8.keys()))
# unique8 = sorted(set(vocab8.keys()) - set(vocab7.keys()))
#
# print(f"✅ Shared tokens: {len(shared_tokens)}")
# print(f"❌ Unique to v7: {len(unique7)}")
# print(f"❌ Unique to v8: {len(unique8)}")
#
# # === Load model ===
# VOCAB_SIZE = len(vocab8)  # assume we want v8 alignment
# model = ArdorDecoder(VOCAB_SIZE, hidden=384, layers=8, heads=6)
# state = torch.load(MODEL_PATH, map_location="cpu")
# model.load_state_dict(state, strict=False)
# model.eval()
#
# # === Embedding & LM head weights ===
# emb = model.token_embed.weight.detach().cpu().numpy()
# lmh = model.lm_head.weight.detach().cpu().numpy()
#
# def cosine(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
#
# # --- Check shared tokens ---
# print("\n🔎 Checking shared tokens consistency (sample 20):")
# for tok in np.random.choice(shared_tokens, size=20, replace=False):
#     i7 = vocab7[tok]; i8 = vocab8[tok]
#     if i7 < emb.shape[0] and i8 < emb.shape[0]:
#         sim = cosine(emb[i7], emb[i8])
#         print(f"{tok:15s} sim={sim:.4f}")
#
# # --- Check v7-unique tokens (possible artifacts) ---
# print("\n🧪 Probing v7-unique tokens (sample 20):")
# for tok in np.random.choice(unique7, size=20, replace=False):
#     i7 = vocab7[tok]
#     if i7 < emb.shape[0]:
#         vec = emb[i7]
#         # find nearest v8 token
#         sims = [cosine(vec, emb[vocab8[t]]) for t in shared_tokens[:200]]
#         best = shared_tokens[int(np.argmax(sims))]
#         print(f"{tok:15s} → closest v8 match: {best}")
#
# # === Generation probe ===
# def generate(prompt, tokenizer, steps=30):
#     encoded = tokenizer.encode(prompt)
#     ids = torch.tensor([encoded.ids], dtype=torch.long)  # [B, T]
#     out = ids
#     for _ in range(steps):
#         with torch.no_grad():
#             logits = model(out)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, 1)
#         out = torch.cat([out, next_id], dim=1)
#     return tokenizer.decode(out[0].tolist())
#
#
# print("\n📝 Generation comparison:")
# for prompt in ["What is love?", "Explain gravity.", "What is death?"]:
#     print(f"\nPrompt: {prompt}")
#     out7 = generate(prompt, tok7)
#     out8 = generate(prompt, tok8)
#     print(f"v7 decoding → {out7}")
#     print(f"v8 decoding → {out8}")



# teacher_selector.py — choose EMA vs Last (or weight them) by validation ppl
# import math, torch, torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
# from tokenizers import Tokenizer
# import sys
# sys.path.append("../Cerebrum/Cortex")
# from broca_decoder import ArdorDecoder
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOKENIZER = r"./Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v8.json"
# HIDDEN, LAYERS, HEADS, DROPOUT = 384, 8, 6, 0.15
#
# def loader(paths, bs=32):
#     d=[];
#     for p in paths:
#         blk = torch.load(p)                    # [N,T]
#         d.append(TensorDataset(blk[:,:-1], blk[:,1:]))
#     return DataLoader(ConcatDataset(d), batch_size=bs, shuffle=False)
#
# def build(vocab):
#     return ArdorDecoder(vocab, hidden_dim=HIDDEN, num_layers=LAYERS, heads=HEADS, dropout=DROPOUT)
#
# @torch.no_grad()
# def ppl(model, dl):
#     model.eval(); tot=0; n=0; V=None
#     for xb,yb in dl:
#         xb,yb=xb.to(DEVICE), yb.to(DEVICE)
#         logits = model(xb); V = logits.size(-1)
#         loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
#         tot += loss.item(); n += 1
#     return math.exp(tot/max(1,n))
#
# def select_teacher(ema_ckpt, last_ckpt, val_paths, prefer_ensemble_margin=0.01):
#     tok = Tokenizer.from_file(TOKENIZER); V = tok.get_vocab_size()
#     dl = loader(val_paths, bs=32)
#
#     ema = build(V).to(DEVICE); ema.load_state_dict(torch.load(ema_ckpt, map_location=DEVICE))
#     last= build(V).to(DEVICE); last.load_state_dict(torch.load(last_ckpt, map_location=DEVICE))
#     p_ema = ppl(ema, dl); p_last = ppl(last, dl)
#
#     print(f"[teacher] EMA ppl={p_ema:.3f}  Last ppl={p_last:.3f}")
#     if p_ema + prefer_ensemble_margin < p_last:
#         return "ema", (ema_ckpt,)
#     if p_last + prefer_ensemble_margin < p_ema:
#         return "last", (last_ckpt,)
#     # close call → ensemble both
#     return "ensemble", (ema_ckpt, last_ckpt)



# import os, json, torch
#
# src = 'C:/Users/adm/PycharmProjects/ProjectArdor/Cerebrum/Models/Ardor/Ardor_Sigma.pt'  # full ckpt you trained
# dst_dir = r'C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Models\Ardor'  # GUI scans here
# os.makedirs(dst_dir, exist_ok=True)
#
# tok = r'C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\ProjectTokenizer\ardor_tokenizer\tokenizer_v9.json'
#
# ck = torch.load(src, map_location='cpu')
# state = ck['model'] if (isinstance(ck, dict) and 'model' in ck) else ck
#
# dst = os.path.join(dst_dir, 'Ardor_SigmaS.pt')
# torch.save(state, dst)
#
# with open(dst.replace('.pt', '.meta.json'), 'w', encoding='utf-8') as f:
#     json.dump({
#         "tokenizer_path": tok,
#         "hidden": 384,
#         "layers": 8,
#         "heads": 6,
#         "max_len": 2048
#     }, f, indent=2)
#
# print('Wrote:', dst)


import torch, os

IN  = r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Models\Ardor\Ardor_Sigma2_weights.LoRA.MERGED.pt"
OUT = r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Models\Ardor\Ardor_GammaΓ.pt"

sd = torch.load(IN, map_location="cpu")
if "state_dict" in sd:  # handle nested formats
    sd = sd["state_dict"]

# strip torch.compile prefix
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

# make sure a head key exists (your model uses to_vocab; add an alias if missing)
if "to_vocab.weight" not in sd and "lm_head.weight" in sd:
    sd["to_vocab.weight"] = sd["lm_head.weight"]
if "lm_head.weight" not in sd and "to_vocab.weight" in sd:
    sd["lm_head.weight"] = sd["to_vocab.weight"]

# sanity prints
print("fixed has token_embed.weight? ", "token_embed.weight" in sd)
print("fixed has to_vocab.weight?    ", "to_vocab.weight" in sd)
print("fixed has lm_head.weight?     ", "lm_head.weight" in sd)

torch.save(sd, OUT)
print("✅ wrote", OUT)

