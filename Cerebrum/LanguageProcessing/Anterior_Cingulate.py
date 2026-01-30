# Anterior_Cingulate.py — output polish & self-monitor (ACC)
import re, string

# --------- resources (stopwords & abbreviations) ----------
try:
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("english"))
except Exception:
    _STOP = {
        "the","a","an","and","or","but","if","then","so","because","as",
        "of","in","on","for","to","from","by","with","about","into","over",
        "is","are","was","were","be","been","being","it","this","that","etc"
    }

_ABBR = {"e.g.", "i.e.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "u.s.", "u.k.", "etc."}

# --------- protect code/urls before editing ----------
_CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
_URL_RE  = re.compile(r"https?://\S+")
def _protect_segments(txt: str):
    vault, i = {}, 0
    def _stash(m):
        nonlocal i
        k = f"@@SEG{i}@@"
        vault[k] = m.group(0)
        i += 1
        return k
    txt = _CODE_RE.sub(_stash, txt)
    txt = _URL_RE.sub(_stash, txt)
    return txt, vault

def _restore_segments(txt: str, vault: dict) -> str:
    for k, v in vault.items():
        txt = txt.replace(k, v)
    return txt

# --------- whitespace & unicode tidy ----------
def _normalize_ws(txt: str) -> str:
    txt = txt.replace("\u00A0", " ").replace("\u2009", " ")
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# --------- repetition breakers ----------
_re_word_loop   = re.compile(r"\b(\w{2,})(?:\s+\1\b)+", re.I)

def _kill_simple_loops(txt: str) -> str:
    # quick pass for "word word word"
    return _re_word_loop.sub(r"\1", txt)

def _dedupe_ngrams(text: str, n_max: int = 5) -> str:
    # token-level: collapses A B C A B C A B → A B C
    toks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    out, i, N = [], 0, len(toks)
    while i < N:
        dedup = False
        # try longer n first
        for n in range(min(n_max, (N - i) // 2), 1, -1):
            if toks[i:i+n] == toks[i+n:i+2*n]:
                # keep one copy, skip the repeats
                out.extend(toks[i:i+n])
                i += 2*n
                while i + n <= N and toks[i-n:i] == toks[i:i+n]:
                    i += n
                dedup = True
                break
        if not dedup:
            out.append(toks[i]); i += 1
    # rejoin with minimal spacing rules
    buf = []
    for j, t in enumerate(out):
        if j and re.match(r"[\w]", t) and re.match(r"[\w)]", out[j-1]):
            buf.append(" ")
        buf.append(t)
    return "".join(buf)

# --------- trailing filler guard ----------
def _trim_trailing_filler(txt: str) -> str:
    words = txt.strip().split()
    if len(words) < 12:
        return txt
    # look at last two tokens (sans punctuation)
    tail1 = words[-1].strip(string.punctuation).lower()
    tail2 = words[-2].strip(string.punctuation).lower() if len(words) >= 2 else ""
    if (tail1 in _STOP or tail1 in {"etc","et","al"}) and (tail2 in _STOP or not tail2):
        words.pop()
        return " ".join(words)
    return txt

# --------- terminal punctuation (guarded) ----------
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def _needs_terminal_punct(txt: str) -> bool:
    s = txt.rstrip()
    if not s:
        return False
    last = s[-1]
    if last in ".!?":  # already punctuated
        return False
    # bullets / lists / code / quotes / brackets / emoji / URLs
    if s.lstrip().startswith(("-", "*", "•")): return False
    if s.endswith(("\"", "”", "'", "’", ")", "]", "}", "`")): return False
    if _EMOJI_RE.search(s[-2:]): return False
    if _URL_RE.search(s[-120:]): return False
    # abbreviations at end (case-insensitive)
    tail = s.split()[-1].lower()
    if tail in _ABBR: return False
    # allow letters/digits to get a period
    return tail[-1].isalnum()

def _complete_sentence(txt: str) -> str:
    return txt + "." if _needs_terminal_punct(txt) else txt

# --------- small typo/grammar fixes ----------
_COMMON = {
    r"\balot\b": "a lot",
    r"\bteh\b": "the",
    r"\brecieve\b": "receive",
    r"\bseperate\b": "separate",
    r"\boccured\b": "occurred",
    r"\bwich\b": "which",
}
def _typo_fix(txt: str) -> str:
    for pat, rep in _COMMON.items():
        txt = re.sub(pat, rep, txt, flags=re.I)
    # collapse repeated punctuation
    txt = re.sub(r"([.!?]){2,}", r"\1", txt)
    # space before punctuation → remove
    txt = re.sub(r"\s+([,.;:!?])", r"\1", txt)
    # single space after punctuation (when followed by a letter)
    txt = re.sub(r"([,.!?])([A-Za-z])", r"\1 \2", txt)
    # ellipses normalization
    txt = re.sub(r"\.{3,}", "…", txt)
    # multiple spaces → single
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt

# --------- public API ----------
def polish(text: str) -> str:
    """
    Pipeline:
      1) protect code/urls
      2) normalize whitespace
      3) kill simple word loops + n-gram repeats
      4) guarded tail trim
      5) guarded terminal punctuation
      6) typo/spacing fixes
      7) restore protected segments
    """
    try:
        t, vault = _protect_segments(text)
        t = _normalize_ws(t)
        t = _kill_simple_loops(t)
        t = _dedupe_ngrams(t)
        t = _trim_trailing_filler(t)
        t = _complete_sentence(t)
        t = _typo_fix(t)
        t = _restore_segments(t, vault)
        return t.strip()
    except Exception:
        return text

