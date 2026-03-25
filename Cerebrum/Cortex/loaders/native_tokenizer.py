from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple

from tokenizers import Tokenizer


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer_matching_vocab(repo_root: Path, tokenizer_path: Optional[str], want_vocab: int, checkpoint_meta: Optional[dict] = None) -> Tuple[Tokenizer, str]:
    meta = checkpoint_meta or {}
    seen: set[str] = set()
    candidates: list[tuple[str, int, float, str]] = []

    roots = [
        str(repo_root / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer"),
        str(repo_root),
        os.getcwd(),
    ]
    if tokenizer_path:
        roots.insert(0, os.path.dirname(os.path.abspath(tokenizer_path)) or os.getcwd())

    meta_tok = meta.get("tokenizer_path")
    if isinstance(meta_tok, str) and meta_tok:
        roots.insert(0, os.path.dirname(os.path.abspath(meta_tok)) or os.getcwd())

    for r in roots:
        rr = os.path.abspath(r)
        if not os.path.isdir(rr):
            continue
        for patt in ("tokenizer*.json", "tokenizer.json"):
            for p in glob.glob(os.path.join(rr, patt)):
                ap = os.path.abspath(p)
                if ap in seen:
                    continue
                seen.add(ap)
                try:
                    t = Tokenizer.from_file(ap)
                    candidates.append((ap, int(t.get_vocab_size()), os.path.getmtime(ap), sha256_file(ap)))
                except Exception:
                    continue

    def pick_by_path(path: Optional[str]):
        if not path:
            return None
        ap = os.path.abspath(path)
        for c in candidates:
            if c[0] == ap:
                return c
        return None

    meta_sha = str(meta.get("tokenizer_sha256") or meta.get("tokenizer_hash") or "").strip().lower()

    for c in (pick_by_path(tokenizer_path), pick_by_path(meta_tok)):
        if c and c[1] == int(want_vocab):
            return Tokenizer.from_file(c[0]), c[0]

    if meta_sha:
        for ap, vocab, _, sha in candidates:
            if sha.lower() == meta_sha and vocab == int(want_vocab):
                return Tokenizer.from_file(ap), ap

    exact = [c for c in candidates if c[1] == int(want_vocab)]
    if exact:
        exact.sort(key=lambda x: x[2], reverse=True)
        return Tokenizer.from_file(exact[0][0]), exact[0][0]

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return Tokenizer.from_file(candidates[0][0]), candidates[0][0]

    raise FileNotFoundError(f"No tokenizer JSON found for vocab {want_vocab}")


def generic_tokenizer_candidates(repo_root: Path) -> List[str]:
    roots = [
        repo_root / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer",
        repo_root / "ProjectTokenizer" / "ardor_tokenizer",
        Path(os.getcwd()),
    ]
    seen: set[str] = set()
    out: List[str] = []
    for root in roots:
        rr = root.resolve() if root.exists() else root
        if not rr.exists() or not rr.is_dir():
            continue
        for patt in ("tokenizer*.json", "tokenizer.json"):
            for fp in rr.glob(patt):
                ap = str(fp.resolve())
                if ap not in seen:
                    seen.add(ap)
                    out.append(ap)
    out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out
