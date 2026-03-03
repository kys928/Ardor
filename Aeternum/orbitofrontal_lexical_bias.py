from __future__ import annotations
from typing import Dict, Iterable
from .protocols import EmotionState

POSITIVE = ["understand","with you","we can","it's okay","let's try","appreciate","glad","thanks","together","helpful"]
SUPPORT  = ["i hear you","i'm here","that's valid","it makes sense","i'll help","step by step","we'll figure it out"]
CAUTIOUS = ["consider","might","could","perhaps","it may","caution","safest","avoid","risk","uncertain"]
DECLINE  = ["cannot","won't","not able","unsafe","against policy","decline","sorry i can't"]

def _ids_for_phrases(tokenizer, phrases: Iterable[str]) -> Iterable[int]:
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False).ids
        if ids:
            yield ids[0]

def lexical_bias(tokenizer, st: EmotionState) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if st.valence > 0.15:
        for tid in _ids_for_phrases(tokenizer, POSITIVE + SUPPORT):
            out[tid] = out.get(tid, 0.0) + 0.20*st.valence
    risk = st.anxiety + st.uncertainty + 0.5*st.surprise
    if risk > 0.9:
        for tid in _ids_for_phrases(tokenizer, CAUTIOUS):
            out[tid] = out.get(tid, 0.0) + 0.15
    if st.plan == "decline":
        for tid in _ids_for_phrases(tokenizer, DECLINE):
            out[tid] = out.get(tid, 0.0) + 0.20
    return out

# Optional: NRC-VAD style lexicon support
import csv
def load_vad_csv(path: str):
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            w = r["word"].strip().lower()
            V = float(r["Valence"]); A = float(r["Arousal"]); D = float(r.get("Dominance", 0.5))
            lex[w] = (V, A, D)
    return lex

def lexical_bias_from_vad(tokenizer, st: EmotionState, vad_lex: dict, scale=0.2) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not vad_lex: return out
    target_v = (st.valence + 1.0) / 2.0  # map [-1,1] to [0,1]
    target = (target_v, st.arousal, st.dominance)
    for w,(V,A,D) in vad_lex.items():
        score = 1.0 - (abs(target[0]-V)+abs(target[1]-A)+abs(target[2]-D))/3.0
        if score > 0.8:
            ids = tokenizer.encode(w, add_special_tokens=False).ids
            if ids:
                out[ids[0]] = out.get(ids[0], 0.0) + scale*(score-0.8)/0.2
    return out
