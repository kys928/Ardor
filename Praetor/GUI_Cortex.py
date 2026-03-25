# GUI_Cortex.py — Ardor Autonomy HUD (Elodin Retro Orange style + scanline + grid)
from __future__ import annotations
import os, sys, time, json, math, random, subprocess, platform, re, traceback as tb
from threading import Thread, Event, Timer
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from typing import Optional
from pathlib import Path

import sys
from pathlib import Path

# Project root = .../ProjectArdor
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aeternum.AeternumCore import AeternumCore, AeternumConfig



# ---- Ardor root resolution (shared with the EXE) ----
RUNTIME_PATHS = getattr(sys.modules.get('__main__'), 'PATHS', None)

def _ardor_root() -> Path:
    env = os.environ.get('ARDOR_HOME')
    if env:
        try: return Path(env).resolve()
        except Exception: pass
    bases: list[Path] = []
    if getattr(sys, 'frozen', False):
        bases.append(Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)))
        bases.append(Path(sys.executable).parent)
    else:
        here = Path(__file__).resolve()
        bases += [here.parent, here.parent.parent, here.parent.parent.parent]
    for b in bases:
        ah = b / 'ardor_home.txt'
        if ah.exists():
            try: return Path(ah.read_text(encoding='utf-8').strip()).resolve()
            except Exception: pass
    cur = bases[0] if bases else Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / 'Cerebrum').exists() or (cur / 'Praetor').exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parent

ROOT_DIR: Path = (RUNTIME_PATHS.ROOT if RUNTIME_PATHS else _ardor_root()).resolve()
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACTS_REM_DIR = ARTIFACTS_DIR / "rem"
ARTIFACTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_REM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- core import ----------------
sys.path.append("../Cerebrum/Cortex")

try:
    from prefrontal_cortex import get_global_core
except Exception:
    get_global_core = None

# ---------------- intents ----------------
class Intent:
    def __init__(self): self.actions=[]
    def add(self,t,**k): self.actions.append({"type":t,**k}); return self
    def __bool__(self): return bool(self.actions)

class IntentParser:
    re_rem=re.compile(r"\b(rem|sleep)\b",re.I)
    re_train=re.compile(r"\btrain(?!ing)\b",re.I)
    re_temp=re.compile(r"\btemp(?:erature)?\s*(?:to|=)?\s*([0-9]*\.?[0-9]+)",re.I)
    re_topp=re.compile(r"\btop[-_ ]?p\s*(?:to|=)?\s*(0?\.\d+|1(?:\.0)?)",re.I)
    re_rep=re.compile(r"\brep(?:etition)?(?:\s*pen(?:alty)?)?\s*(?:to|=)?\s*([0-9]*\.?[0-9]+)",re.I)
    re_ngram=re.compile(r"\bngram\b.*?(\d+)",re.I)
    re_model=re.compile(r"\b(?:model|switch(?:\s*to)?)\s+([\w_.-]+\.pt)\b",re.I)
    re_delay=re.compile(r"\bin\s+(\d+)\s*(s|sec|seconds|m|min|minutes|h|hours)\b",re.I)
    re_abstime=re.compile(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",re.I)
    @staticmethod
    def _to_seconds(n,u):
        u=u.lower(); return int(n)*(1 if u.startswith('s') else 60 if u.startswith('m') else 3600)
    @staticmethod
    def _seconds_until(hh,mm,ap):
        if ap:
            ap=ap.lower()
            if hh==12: hh=0
            if ap=='pm': hh+=12
        now=datetime.now()
        tgt=now.replace(hour=hh%24, minute=mm%60, second=0, microsecond=0)
        if tgt<=now: tgt+=timedelta(days=1)
        return int((tgt-now).total_seconds())
    def parse(self, text: str) -> Intent:
        it = Intent(); delay = 0
        m_abs = self.re_abstime.search(text)
        if m_abs:
            hh = int(m_abs.group(1)); mm = int(m_abs.group(2) or 0); ap = m_abs.group(3)
            delay = self._seconds_until(hh, mm, ap)
        else:
            m = self.re_delay.search(text)
            if m: delay = self._to_seconds(int(m.group(1)), m.group(2))
        if self.re_rem.search(text):   it.add("rem", delay=delay)
        if self.re_train.search(text): it.add("train")
        m = self.re_temp.search(text)
        if m: it.add("set", key="temperature", value=float(m.group(1)))
        m = self.re_topp.search(text)
        if m: it.add("set", key="top_p", value=float(m.group(1)))
        m = self.re_rep.search(text)
        if m: it.add("set", key="rep_penalty", value=float(m.group(1)))
        m = self.re_ngram.search(text)
        if m: it.add("set", key="ngram", value=int(m.group(1)))
        m = self.re_model.search(text)
        if m: it.add("switch_model", name=m.group(1))
        return it

AUTONOMY_LEVELS=("Off","Assist","Copilot","Autonomous")
ALLOW={
 "Off":        {"auto": set(),                           "confirm":{"set","switch_model","rem","train"}},
 "Assist":     {"auto":{"set","switch_model"},           "confirm":{"rem","train"}},
 "Copilot":    {"auto":{"set","switch_model","rem"},     "confirm":{"train"}},
 "Autonomous": {"auto":{"set","switch_model","rem"},     "confirm":{"train"}},
}

STYLE_PRIME = (
    "You are Ardor. Answer in clear, natural, conversational English. "
)

# ───────────────────────── Atelier-style Retro Theme ─────────────────────────
RETRO_DARK  = {'bg':'#0a0907','text':'#fff4e2','panel':'#100e0b','panel2':'#13100b',
               'stroke':'#3b2a1d','accent1':'#ffc861','accent2':'#ffb354','accent3':'#ff9100','accent4':'#d46b00','grid':'#2f2216'}
RETRO_LIGHT = {'bg':'#fff8ed','text':'#2b1a0d','panel':'#fff2e0','panel2':'#ffe9cc',
               'stroke':'#e6cfb1','accent1':'#ffb354','accent2':'#ffc861','accent3':'#d46b00','accent4':'#a65700','grid':'#f2d8b8'}

# ---------------- color helpers ----------------
def _hex_to_rgb(h: str):
    h = (h or "").lstrip('#');
    if len(h) != 6: return (255, 255, 255)
    return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0,min(255,int(v))) for v in rgb)
def _lerp(a,b,t): return a+(b-a)*t
def _mix(c1:str, c2:str, t:float)->str:
    r1,g1,b1=_hex_to_rgb(c1); r2,g2,b2=_hex_to_rgb(c2)
    return _rgb_to_hex((_lerp(r1,r2,t), _lerp(g1,g2,t), _lerp(b1,b2,t)))

# ---------------- Scanline bar (retro) ----------------
class Scanline(tk.Canvas):
    def __init__(self, master, theme, **kw):
        super().__init__(master, height=3, highlightthickness=0, **kw)
        self.theme = theme
        self._x = -100
        self.configure(bg=self.theme['panel'])
        self._tick()

    def set_theme(self, theme):
        self.theme = theme
        self.configure(bg=self.theme['panel'])

    def _tick(self):
        self.delete('all')
        w = self.winfo_width() or 1
        h = self.winfo_height() or 3
        x0 = self._x - 80
        self.create_rectangle(x0, 0, x0+80, h, fill=self.theme['accent3'], outline="")
        self.create_rectangle(x0+80, 0, x0+96, h, fill=self.theme['accent1'], outline="")
        self._x = (self._x + 6) % (w + 160)
        self.after(24, self._tick)

# ---------------- Kaomoji HUD (Canvas grid + centered face) ----------------

# 28 labels you requested
KAOMOJI_LABELS = [
    "Admiration","Amusement","Anger","Annoyance","Approval","Caring","Confusion","Curiosity",
    "Desire","Disappointment","Disapproval","Disgust","Embarrassment","Excitement","Fear","Gratitude",
    "Grief","Joy","Love","Nervousness","Optimism","Pride","Realization","Relief","Remorse","Sadness",
    "Surprise","Neutral"
]

# Valence ∈ [-1,1], Arousal ∈ [0,1]
MOOD_META = {
    "Admiration":     ( 0.7, 0.6),
    "Amusement":      ( 0.6, 0.5),
    "Anger":          (-0.7, 0.9),
    "Annoyance":      (-0.3, 0.6),
    "Approval":       ( 0.6, 0.5),
    "Caring":         ( 0.8, 0.4),
    "Confusion":      (-0.1, 0.5),
    "Curiosity":      ( 0.3, 0.6),
    "Desire":         ( 0.6, 0.7),
    "Disappointment": (-0.5, 0.4),
    "Disapproval":    (-0.5, 0.6),
    "Disgust":        (-0.7, 0.5),
    "Embarrassment":  (-0.3, 0.6),
    "Excitement":     ( 0.9, 0.9),
    "Fear":           (-0.6, 0.9),
    "Gratitude":      ( 0.8, 0.5),
    "Grief":          (-0.9, 0.7),
    "Joy":            ( 0.8, 0.6),
    "Love":           ( 0.9, 0.7),
    "Nervousness":    (-0.2, 0.7),
    "Optimism":       ( 0.7, 0.5),
    "Pride":          ( 0.6, 0.5),
    "Realization":    ( 0.2, 0.5),
    "Relief":         ( 0.6, 0.3),
    "Remorse":        (-0.6, 0.5),
    "Sadness":        (-0.8, 0.4),
    "Surprise":       ( 0.0, 0.9),
    "Neutral":        ( 0.0, 0.2),
}

# Single-line kaomoji per label
KAOMOJI = {
    "Admiration":     "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧",
    "Amusement":      "(＾▽＾)",
    "Anger":          "(╬ಠ益ಠ)",
    "Annoyance":      "(¬_¬ )",
    "Approval":       "(b'_')b",
    "Caring":         "(つ˘◡˘)つ",
    "Confusion":      "(・・ )?",
    "Curiosity":      "(⊙_⊙)?",
    "Desire":         "(♡ω♡ )",
    "Disappointment": "(︶︹︺)",
    "Disapproval":    "(ಠ_ಠ)",
    "Disgust":        "(¬､¬)",
    "Embarrassment":  "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)",
    "Excitement":     "ヽ(＾Д＾)ﾉ",
    "Fear":           "(ó﹏ò｡)",
    "Gratitude":      "(人❛ᴗ❛)♪",
    "Grief":          "(٭°̧̧̧ω°̧̧̧٭)",
    "Joy":            "(＾▽＾)",
    "Love":           "(♡˙︶˙♡)",
    "Nervousness":    "(；・∀・)",
    "Optimism":       "(＾∇＾)",
    "Pride":          "(￣‿￣)",
    "Realization":    "(・o・)",
    "Relief":         "(〃´o｀)=3",
    "Remorse":        "(シ_ _)シ",
    "Sadness":        "(╥_╥)",
    "Surprise":       "Σ(°ロ°)",
    "Neutral":        "(・_・)"
}

KAOMOJI_ROTATE_SECS = 15  # ← change this if you want a different rotation interval

def _mood_color(theme, valence: float, arousal: float) -> str:
    """Warm (orange) for positive valence, cool (teal) for negative; brighten with arousal."""
    warm = theme.get('accent3', '#ff9100')
    cool = '#3aa7b8'
    t = (valence * 0.5) + 0.5  # [-1,1] -> [0,1]
    base = _mix(cool, warm, t)
    return _mix(base, "#ffffff", 0.12 + 0.30*max(0.0, min(1.0, arousal)))

class HUD(tk.Canvas):
    """Minimal, grid-based kaomoji HUD. Keeps method names so the rest of GUI doesn't break."""
    def __init__(self, master, theme, **kw):
        super().__init__(master, highlightthickness=0, **kw)
        self.theme = theme
        self.configure(bg=self.theme['bg'])
        self._resize_job = None
        self.current_label = random.choice(KAOMOJI_LABELS)
        self.face_id = None
        self.label_id = None
        self.dot_id = None

        # fonts (Tk will fall back if not available)
        self.face_font = ("Segoe UI Emoji", 48, "normal")
        self.label_font = ("Consolas", 14, "normal")

        self.bind("<Configure>", self._on_configure)
        self._draw_static()
        self._schedule_rotate()

    # Compatibility shims
    def set_theme(self, theme):
        self.theme = theme or RETRO_DARK
        self.configure(bg=self.theme['bg'])
        self._rebuild_static()

    def look_at_root(self, _x_root: int, _y_root: int):
        # Kept for compatibility; not used in the kaomoji HUD
        pass

    def bias_towards_widget(self, _widget: tk.Widget, _duration: float = 1.5):
        # Kept for compatibility; not used here
        pass

    def set_status(self, _status: str, _revert_after: float|None=None):
        # Kept for compatibility
        pass

    # Drawing
    def _on_configure(self, _evt):
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self._rebuild_static)

    def _rebuild_static(self):
        self._resize_job = None
        self.delete('all')
        self.face_id = self.label_id = self.dot_id = None
        self._draw_static()

    def _draw_grid(self):
        grid_c = self.theme.get('grid', '#2f2216')
        w = self.winfo_width() or int(float(self.cget('width') or 0)) or 1
        h = self.winfo_height() or int(float(self.cget('height') or 0)) or 1
        step = 28
        for x in range(0, w, step):
            self.create_line(x, 0, x, h, fill=grid_c)
        for y in range(0, h, step):
            self.create_line(0, y, w, y, fill=grid_c)

    def _draw_static(self):
        self._draw_grid()
        self._render_face()

    def set_from_emotion_state(self, st):
        """
        Map EmotionState to one of the KAOMOJI_LABELS.
        We use anxiety/valence/arousal/stance to pick a label.
        """
        if st is None:
            return

        # Safe getters (in case the dataclass changes)
        val   = float(getattr(st, "valence", 0.0))
        aro   = float(getattr(st, "arousal", 0.0))
        anx   = float(getattr(st, "anxiety", 0.0))
        surpr = float(getattr(st, "surprise", 0.0))
        stance = getattr(st, "stance", "") or ""

        label = "Neutral"

        # High threat → fear / nervous
        if anx > 0.7 or stance == "cautious":
            label = "Fear"
        elif anx > 0.4:
            label = "Nervousness"

        # Strong positive, not too aroused → caring / gratitude / love
        elif val > 0.4 and aro < 0.6:
            if stance == "supportive":
                label = "Caring"
            else:
                label = "Gratitude"

        # High positive + high arousal → joy / excitement
        elif val > 0.5 and aro >= 0.6:
            label = "Excitement" if surpr < 0.4 else "Surprise"

        # Negative valence, low arousal → sadness
        elif val < -0.5 and aro < 0.6:
            label = "Sadness"

        # Negative valence, high arousal → anger / annoyance
        elif val < -0.4 and aro >= 0.6:
            label = "Anger" if getattr(st, "dominance", 0.0) > 0.5 else "Annoyance"

        self.current_label = label
        self._render_face()


    def _render_face(self):
        w = self.winfo_width() or int(float(self.cget('width') or 1))
        h = self.winfo_height() or int(float(self.cget('height') or 1))
        cx, cy = w//2, h//2

        label = self.current_label
        face = KAOMOJI.get(label, "(・_・)")
        val, aro = MOOD_META.get(label, (0.0, 0.2))
        color = _mood_color(self.theme, val, aro)

        # Fit face size to viewport
        # Scale font based on width (rough heuristic)
        face_size = max(28, min(96, int(0.07 * min(w, h) + 36)))
        self.face_font = ("Segoe UI Emoji", face_size, "normal")

        # draw colored dot + label under face
        label_y = cy + int(face_size * 0.9)

        # face
        if self.face_id is None:
            self.face_id = self.create_text(cx, cy, text=face, fill=color, font=self.face_font, anchor="center")
        else:
            self.coords(self.face_id, cx, cy)
            self.itemconfigure(self.face_id, text=face, fill=color, font=self.face_font)

        # label
        label_text = f"{label}   ·   valence {val:+.2f} | arousal {aro:.2f}"
        if self.label_id is None:
            self.label_id = self.create_text(cx, label_y, text=label_text,
                                             fill=self.theme['text'], font=self.label_font, anchor="center")
        else:
            self.coords(self.label_id, cx, label_y)
            self.itemconfigure(self.label_id, text=label_text, fill=self.theme['text'], font=self.label_font)

        # small colored dot next to label (left side)
        dot_r = 6
        dot_x = cx - (self.bbox(self.label_id)[2] - self.bbox(self.label_id)[0])//2 - 16
        dot_y = label_y
        if self.dot_id is None:
            self.dot_id = self.create_oval(dot_x-dot_r, dot_y-dot_r, dot_x+dot_r, dot_y+dot_r,
                                           fill=color, outline="")
        else:
            self.coords(self.dot_id, dot_x-dot_r, dot_y-dot_r, dot_x+dot_r, dot_y+dot_r)
            self.itemconfigure(self.dot_id, fill=color)

    # rotation
    def _schedule_rotate(self):
        # self.after(KAOMOJI_ROTATE_SECS * 1000, self._rotate_once)
        pass
    def _rotate_once(self):
        # pick a different mood than current
        options = [m for m in KAOMOJI_LABELS if m != self.current_label]
        if options:
            self.current_label = random.choice(options)
            self._render_face()
        self._schedule_rotate()

# ---------------- GUI ----------------
DECODE_CFG = {"temperature": 0.70, "top_p": 0.90, "rep_penalty": 1.15, "ngram": 4}

WIN_MODELS_PATH = Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Models\Ardor")
if (ARTIFACTS_MODELS_DIR).exists():
    MODELS_DIR = str(ARTIFACTS_MODELS_DIR.resolve())
elif WIN_MODELS_PATH.exists():
    MODELS_DIR = str(WIN_MODELS_PATH.resolve())
else:
    MODELS_DIR = str((ROOT_DIR / "Cerebrum" / "Models" / "Ardor").resolve())

WIN=platform.system()=="Windows"

class ArdorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Ardor")
        sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
        W=min(1280,int(sw*0.96)); H=min(860,int(sh*0.92))
        self.geometry(f"{W}x{H}"); self.minsize(980,640); self.resizable(True,True)

        # Atelier vibe by default
        self.theme=RETRO_DARK
        self.parser=IntentParser(); self.core=None

        # Aeternum (emotion core)
        try:
            cfg = AeternumConfig(device="cpu", hidden_dim=384, prefer_snn=True)
            self.aeternum = AeternumCore(cfg)
            self.last_emotion = None
            self.log("[Aeternum] Emotion core loaded (vmPFC + Amygdala SNN).", tag='sys')
        except Exception as e:
            self.aeternum = None
            self.last_emotion = None
            self.log(f"[Aeternum] Failed to init emotion core: {e}", tag='sys')


        self.current_model=None; self.proc=None; self.anim_event=Event()
        self.pending_plan=None; self.autonomy=tk.StringVar(value="Assist")

        # Slow-type state
        self._slowtype_job=None
        self._slowtype_cancel=False
        self._slowtype_cps=70.0     # characters per second (tweak here)
        self._slowtype_chunk=3      # characters per tick

        # Top bar
        self.top=tk.Frame(self,bg=self.theme['panel'],bd=0,highlightthickness=0)
        self.top.pack(fill=tk.X, padx=12, pady=(10,8))

        tk.Label(self.top,text="Autonomy:",fg=self.theme['text'],bg=self.theme['panel']).pack(side=tk.LEFT,padx=(6,6))
        self.autonomy_box=ttk.Combobox(self.top, values=AUTONOMY_LEVELS, textvariable=self.autonomy, state="readonly", width=12)
        self.autonomy_box.pack(side=tk.LEFT, padx=(0,10))

        # Models dropdown (select to load)
        tk.Label(self.top, text="Model:", fg=self.theme['text'], bg=self.theme['panel']).pack(side=tk.LEFT, padx=(4,4))
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(self.top, textvariable=self.model_var, state="readonly", width=34)
        self.model_box.pack(side=tk.LEFT)
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.on_select_model())

        # Tokenizers dropdown (select to load)
        tk.Label(self.top, text="Tokenizer:", fg=self.theme['text'], bg=self.theme['panel']).pack(side=tk.LEFT, padx=(10,4))
        self.tokenizer_var = tk.StringVar()
        self.tokenizer_box = ttk.Combobox(self.top, textvariable=self.tokenizer_var, state="readonly", width=38)
        self.tokenizer_box.pack(side=tk.LEFT)
        self.tokenizer_box.bind("<<ComboboxSelected>>", lambda e: self.on_select_tokenizer())

        # Atelier launcher + theme toggle (right) — themed button
        self.atelier_btn = tk.Button(self.top, text="Atelier (Editor)",
                                     command=self.switch_to_atelier,
                                     bg=self.theme['panel'], fg=self.theme['text'],
                                     activebackground=self.theme['accent1'],
                                     relief=tk.RAISED, bd=1, highlightthickness=0)
        self.atelier_btn.pack(side=tk.RIGHT, padx=6)

        self.theme_btn=tk.Button(self.top,text="Theme", command=self.toggle_theme,
                                 bg=self.theme['panel'], fg=self.theme['text'],highlightthickness=0, bd=1)
        self.theme_btn.pack(side=tk.RIGHT,padx=(6,0))

        # Scanline strip (retro accent)
        self.scanline = Scanline(self, theme=self.theme)
        self.scanline.pack(fill=tk.X, padx=12)

        # Panes
        self.panes = tk.PanedWindow(self, orient=tk.VERTICAL, sashwidth=6, bg=self.theme['panel'], bd=0, relief=tk.FLAT)
        self.panes.pack(fill=tk.BOTH, expand=True)

        # HUD (now kaomoji canvas)
        self.hud_frame = tk.Frame(self.panes, bg=self.theme['bg'])
        self.hud = HUD(self.hud_frame, theme=self.theme, width=W-24, height=int(H*0.40))
        self.hud.pack(fill=tk.BOTH, expand=True, padx=12)
        self.panes.add(self.hud_frame, minsize=140)

        # Output
        self.trans_frame = tk.Frame(self.panes, bg=self.theme['bg'])
        self.term = tk.Text(self.trans_frame, bg=self.theme['bg'], fg=self.theme['text'],
                            insertbackground=self.theme['accent3'], font=("Consolas",11),
                            height=16, wrap='word', relief=tk.FLAT, bd=1, highlightthickness=1, highlightbackground=self.theme['stroke'])
        self.term.pack(fill=tk.BOTH, expand=True, padx=12)
        self._enable_mousewheel(self.term)
        self.panes.add(self.trans_frame, minsize=120)

        # Input
        self.input_frame = tk.Frame(self.panes, bg=self.theme['bg'])
        self.input = tk.Text(self.input_frame, height=4, wrap='word', font=("Consolas",11),
                             bg=self.theme['panel2'], fg=self.theme['text'],
                             insertbackground=self.theme['accent3'], bd=1,
                             highlightthickness=1, highlightbackground=self.theme['stroke'])
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12,8), pady=(6,10))
        self.input.bind('<KeyRelease>', self._autosize_input)
        self.input.bind('<Return>', self._on_return)
        self.input.bind('<Shift-Return>', lambda e: None)
        self.input.bind('<Key>', self._on_any_key)

        self.send_btn = tk.Button(self.input_frame, text="Send", command=self.on_send,
                                  bg=self.theme['accent3'], fg='#000', activebackground=self.theme['accent1'],
                                  relief=tk.RAISED, bd=1, highlightthickness=0)
        self.send_btn.pack(side=tk.RIGHT, padx=12, pady=(6,10))
        self.panes.add(self.input_frame, minsize=64)

        self.after(120, self._init_sashes)
        self.bind_all('<Motion>', self._on_motion)

        # Progress/status
        self.progress = ttk.Progressbar(self, mode='determinate', length=380)
        self.status = tk.Label(self, text="REM: Idle", fg=self.theme['text'], bg=self.theme['bg'])

        # NEW: configure text tags for colored output
        self._configure_text_tags()

        # Model & Tokenizer boot
        self.update_model_dropdown()
        self.update_tokenizer_dropdown()
        latest=self.find_latest_model()
        if latest:
            tok_path = self.get_selected_tokenizer_path()
            self.log(f"Loading latest model…", tag='sys')
            self.load_model(latest, tok_override=tok_path)
        else:
            self.log(f"⚠️ No models found in {MODELS_DIR}", tag='sys')

        Thread(target=self.simulate_rem_sleep, daemon=True).start()
        Thread(target=self.watch_new_model, daemon=True).start()

        self.bind('<F1>', lambda e: self.run_tests())
        self.bind('<F5>', lambda e: (self.update_model_dropdown(), self.update_tokenizer_dropdown()))
        self.set_theme(self.theme)
        self.after(400, lambda: self.hud.bias_towards_widget(self.input, _duration=1.2))
        self.log("[boot] Ardor GUI ready.\n", tag='sys')

    # ----- UI helpers to ensure all widget updates happen on the Tk main thread
    def ui(self, fn, *args, **kwargs):
        self.after(0, lambda: fn(*args, **kwargs))

    def ui_log(self, msg: str, nl: bool = False, tag: str = 'sys'):
        self.ui(self.log, msg, nl, tag)

    # ----- colored text helpers -----
    def _configure_text_tags(self):
        self.term.tag_configure('sys', foreground=self.theme['accent3'])
        dark_bg = (self.theme.get('bg') == RETRO_DARK.get('bg'))
        ardor_fg = '#ffffff' if dark_bg else '#000000'
        self.term.tag_configure('ardor', foreground=ardor_fg)
        self.term.tag_configure('default', foreground=self.theme['text'])
        self.term.tag_configure('thinking', foreground=ardor_fg)

    def update_emotion_from_text(self, text: str):
        """Run user text through Aeternum + SNN and update HUD."""
        if not getattr(self, "aeternum", None):
            return
        pooled = None

        # try to reuse Ardor's encoder as pooled embedding source
        # try:
        #     if getattr(self, "core", None) is not None and hasattr(self.core, "parietal"):
        #         # _encode returns a 1D torch.Tensor on the same device
        #         pooled = self.core.parietal._encode(text)
        # except Exception as e:
        #     self.log(f"[Aeternum] pooling failed: {e}", tag='sys')
        #
        # try:
        #     dec = self.aeternum.update(
        #         text=text,
        #         pooled_embedding=pooled,
        #         is_new_turn=True,
        #     )
        #     st = dec.state
        #     self.last_emotion = st
        #     # drive kaomoji HUD
        #     self.hud.set_from_emotion_state(st)
        # except Exception as e:
        #     self.log(f"[Aeternum] update failed: {e}", tag='sys')

    def _append(self, text: str, tag: str = 'default'):
        # During very early init, self.term doesn't exist yet.
        term = getattr(self, "term", None)
        if term is None:
            # Fallback: just print to console so we still see the boot logs.
            print(text, end="")
            return

        start = term.index('end-1c')
        term.insert(tk.END, text)
        end = term.index('end-1c')
        try:
            term.tag_add(tag, start, end)
        except Exception:
            pass
        term.see(tk.END)

    def _append_chunk(self, chunk: str, tag: str):
        """Append without extra logic; used by slow-typing."""
        self.term.insert(tk.END, chunk)
        end = self.term.index('end-1c')
        start = f"{float(end) - 0.0}c"
        try:
            self.term.tag_add(tag, f"{end} - {len(chunk)}c", end)
        except Exception:
            pass
        self.term.see(tk.END)

    def log(self, msg:str, nl:bool=False, tag:str='sys'):
        self._append(msg + ('' if nl else '\n'), tag=tag)

    def show_thinking(self):
        self.stop_slow_type()
        try:
            rng = self.term.tag_nextrange('thinking', '1.0')
            while rng:
                self.term.delete(rng[0], rng[1])
                rng = self.term.tag_nextrange('thinking', rng[0])
        except Exception:
            pass
        if self.term.get('end-2c', 'end-1c') != '\n':
            self._append("\n", tag='ardor')
        start_prefix = self.term.index('end-1c')
        self.term.insert(tk.END, "🧠 Ardor: ")
        self.term.tag_add('ardor', start_prefix, self.term.index('end-1c'))
        think_start = self.term.index('end-1c')
        self.term.insert(tk.END, "Thinking…")
        think_end = self.term.index('end-1c')
        self.term.tag_add('ardor', think_start, think_end)
        self.term.tag_add('thinking', think_start, think_end)
        self.term.see(tk.END)

    def replace_thinking_with_stream(self, text: str, tag: str = 'ardor'):
        rng = self.term.tag_nextrange('thinking', '1.0')
        if not rng:
            self.slow_type((text if text.endswith("\n") else text + "\n"), tag=tag)
            return
        start, end = rng
        self.term.delete(start, end)
        mark = f"st_{int(time.time() * 1000)}"
        self.term.mark_set(mark, start)
        self.term.mark_gravity(mark, tk.RIGHT)
        self.slow_type((text if text.endswith("\n") else text + "\n"), tag=tag, at_mark=mark)

    # ----- slow-typing -----
    def stop_slow_type(self):
        self._slowtype_cancel = True
        if self._slowtype_job:
            try:
                self.after_cancel(self._slowtype_job)
            except Exception:
                pass
        self._slowtype_job = None

    def slow_type(self, text: str, tag: str = 'ardor', cps: float | None = None,
                  chunk: int | None = None, trailing_newline: bool = True,
                  at_mark: str | None = None):
        self.stop_slow_type()
        cps = float(self._slowtype_cps if cps is None else cps)
        chunk = int(self._slowtype_chunk if chunk is None else chunk)
        text = text if text.endswith("\n") or not trailing_newline else (text + "\n")
        delay_ms = max(5, int(1000.0 * (chunk / max(1.0, cps))))
        self._slowtype_cancel = False
        i = 0
        N = len(text)

        def insert_chunk_at_mark(mark: str, chunk_text: str):
            start = self.term.index(mark)
            self.term.insert(mark, chunk_text)
            end = self.term.index(mark)
            try:
                self.term.tag_add(tag, start, end)
            except Exception:
                pass
            self.term.see(mark)

        def step():
            nonlocal i
            if self._slowtype_cancel:
                remain = text[i:]
                if remain:
                    if at_mark:
                        insert_chunk_at_mark(at_mark, remain)
                    else:
                        self._append_chunk(remain, tag)
                self._slowtype_job = None
                return
            j = min(N, i + chunk)
            piece = text[i:j]
            if at_mark:
                insert_chunk_at_mark(at_mark, piece)
            else:
                self._append_chunk(piece, tag)
            i = j
            if i >= N:
                self._slowtype_job = None
                return
            self._slowtype_job = self.after(delay_ms, step)

        if not at_mark and self.term.get('end-2c', 'end-1c') != '\n':
            self._append("\n", tag=tag)
        self._slowtype_job = self.after(0, step)

    # ----- events
    def _on_motion(self, e): self.hud.look_at_root(e.x_root, e.y_root)
    def _on_any_key(self, _e=None): self.hud.bias_towards_widget(self.input, _duration=1.5)

    # ----- panes helpers
    def _init_sashes(self):
        try:
            H=self.panes.winfo_height()
            self.panes.sashpos(0, int(H*0.42))
            self.panes.sashpos(1, int(H*0.78))
        except Exception:
            pass

    def _enable_mousewheel(self, widget: tk.Text):
        widget.bind("<MouseWheel>", lambda e: widget.yview_scroll(-1*(e.delta//120), "units"))
        widget.bind("<Button-4>",  lambda e: widget.yview_scroll(-1, "units"))
        widget.bind("<Button-5>",  lambda e: widget.yview_scroll(+1, "units"))

    # ----- input UX
    def _autosize_input(self, _=None):
        try:
            lines=int(self.input.index('end-1c').split('.')[0])
            want=min(max(lines, 4), 7)
            if want!=int(self.input.cget('height')):
                self.input.configure(height=want)
        except Exception:
            pass

    def _on_return(self, e):
        if e.state & 0x0001:  # Shift inserts newline
            return
        self.on_send(); return 'break'

    # ----- theme
    def toggle_theme(self):
        self.set_theme(RETRO_LIGHT if self.theme is RETRO_DARK else RETRO_DARK)

    def set_theme(self, theme):
        self.theme=theme
        self.configure(bg=theme['bg'])
        self.top.configure(bg=theme['panel'])
        for w in self.top.winfo_children():
            if isinstance(w, (tk.Label, tk.Button)):
                w.configure(bg=theme['panel'], fg=theme['text'])
        self.panes.configure(bg=theme['panel'])
        for f in (self.hud_frame, self.trans_frame, self.input_frame):
            f.configure(bg=theme['bg'])

        self.term.configure(bg=theme['bg'], fg=theme['text'], insertbackground=self.theme['accent3'],
                            highlightbackground=self.theme['stroke'])
        self.status.configure(bg=theme['bg'], fg=theme['text'])
        self.input.configure(bg=theme['panel2'], fg=theme['text'],
                             insertbackground=self.theme['accent3'], highlightbackground=self.theme['stroke'])
        self.send_btn.configure(bg=self.theme['accent3'], activebackground=self.theme['accent1'])
        self.theme_btn.configure(bg=theme['panel'], fg=theme['text'])
        self.atelier_btn.configure(bg=theme['panel'], fg=theme['text'], activebackground=self.theme['accent1'])
        self.scanline.set_theme(theme)
        self.hud.set_theme(theme)
        self._configure_text_tags()

    # ----- model mgmt
    def list_models(self):
        dirs = [Path(MODELS_DIR), ROOT_DIR / "Cerebrum" / "Models" / "Ardor"]
        seen = {}
        for d in dirs:
            try:
                if not d.is_dir():
                    continue
                for p in d.glob("*.pt"):
                    seen[p.name] = str(p.resolve())
            except Exception:
                continue
        names = list(seen.keys())
        names.sort(key=lambda n: os.path.getmtime(seen[n]), reverse=True)
        self._model_name_to_path = seen
        return names

    def update_model_dropdown(self):
        models=self.list_models()
        self.model_box['values']=models
        want = self.current_model if self.current_model in models else (models[0] if models else "")
        self.model_var.set(want)

    def on_select_model(self):
        name = self.model_var.get().strip()
        if not name: return
        path=getattr(self, "_model_name_to_path", {}).get(name, os.path.join(MODELS_DIR, name))
        if not os.path.isfile(path):
            self.log(f"⚠️ Not found: {path}", tag='sys'); return
        tok_path = self.get_selected_tokenizer_path()
        self.load_model(path, tok_override=tok_path)

    def find_latest_model(self):
        d=MODELS_DIR
        if not os.path.isdir(d): return None
        pts=[]
        for lookup in [Path(d), ROOT_DIR / "Cerebrum" / "Models" / "Ardor"]:
            if lookup.is_dir():
                pts.extend(str(p.resolve()) for p in lookup.glob("*.pt"))
        return max(pts, key=lambda p: os.path.getmtime(p)) if pts else None

    def resolve_tokenizer_path(self):
        cand = [
            ROOT_DIR / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v9.json",
            ROOT_DIR / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json",
            ROOT_DIR / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v7.json",
            ROOT_DIR / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v6.json",
            ROOT_DIR / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v9.json",
            ROOT_DIR / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v8.json",
            ROOT_DIR / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v7.json",
            ROOT_DIR / "ProjectTokenizer" / "ardor_tokenizer" / "tokenizer_v6.json",
        ]
        for p in cand:
            if p.is_file(): return str(p.resolve())
        return None

    def _model_vocab_size(self, model) -> Optional[int]:
        cand = ["token_embed","tok_emb","embed","embedding","wte","embeddings","token_embedding"]
        for name in cand:
            w = getattr(model, name, None)
            if w is None: continue
            if hasattr(w, "weight"):
                try: return int(w.weight.size(0))
                except Exception: pass
            try: return int(w.size(0))
            except Exception: pass
        return None

    # ----- tokenizer mgmt -----
    def list_tokenizers(self):
        roots = []
        env_dir = os.environ.get("ARDOR_TOKENIZER_DIR")
        if env_dir and os.path.isdir(env_dir):
            roots.append(Path(env_dir))
        roots += [
            ROOT_DIR / "Cerebrum" / "ProjectTokenizer" / "ardor_tokenizer",
            ROOT_DIR / "ProjectTokenizer" / "ardor_tokenizer",
        ]
        roots.append(Path(r"C:\\Users\\adm\\PycharmProjects\\ProjectArdor\\Cerebrum\\ProjectTokenizer\\ardor_tokenizer"))

        seen = set(); items = []
        for r in roots:
            try:
                for p in Path(r).glob("tokenizer_v*.json"):
                    fp = str(p.resolve())
                    if fp not in seen:
                        seen.add(fp); items.append(fp)
            except Exception:
                pass

        def vernum(p):
            m = re.search(r"tokenizer_v(\d+)\.json", os.path.basename(p))
            return int(m.group(1)) if m else -1
        items.sort(key=vernum, reverse=True)

        labels = ["Auto (match model vocab)"] + [os.path.basename(p) for p in items]
        mapping = {"Auto (match model vocab)": ""} | {os.path.basename(p): p for p in items}
        return labels, mapping

    def update_tokenizer_dropdown(self):
        labels, mapping = self.list_tokenizers()
        self._tok_label_to_path = mapping
        self.tokenizer_box['values'] = labels
        if getattr(self, "core", None) and getattr(self.core, "tokenizer_path", None):
            used = os.path.basename(self.core.tokenizer_path)
            if used in mapping:
                self.tokenizer_var.set(used); return
        self.tokenizer_var.set("Auto (match model vocab)")

    def get_selected_tokenizer_path(self) -> str:
        label = (self.tokenizer_var.get() or "").strip()
        if not hasattr(self, "_tok_label_to_path"):
            self.update_tokenizer_dropdown()
        return self._tok_label_to_path.get(label, "")

    def on_select_tokenizer(self):
        if not self.current_model:
            self.log("⚠️ No model is loaded yet.", tag='sys'); return
        tok_path = self.get_selected_tokenizer_path()
        self.load_model(os.path.join(MODELS_DIR, self.current_model), tok_override=tok_path)

    # ----- load/switch -----
    def load_model(self, path, tok_override: Optional[str] = None) -> bool:
        try:
            meta_path = os.path.splitext(path)[0] + ".meta.json"
            tok = ""
            if isinstance(tok_override, str):
                tok = tok_override
            elif os.path.isfile(meta_path):
                try:
                    meta = json.load(open(meta_path, "r", encoding="utf-8"))
                    tok_meta = meta.get("tokenizer_path") or ""
                    if tok_meta and os.path.isfile(tok_meta):
                        tok = tok_meta
                    elif tok_meta:
                        self.log(f"⚠️ Tokenizer from meta not found at {tok_meta}; will auto-select.", tag='sys')
                except Exception as e:
                    self.log(f"⚠️ Failed reading meta: {e}", tag='sys')

            if not callable(get_global_core):
                self.log("❌ get_global_core entrypoint is unavailable.", tag='sys'); return False

            try:
                self.core = get_global_core(model_path=path, tokenizer_path=tok if tok and os.path.isfile(tok) else None, device='cpu', enable_retrieval=True, encoder_ckpt=None, max_len=getattr(self, 'max_len', 300), force_reload=True)
                desc = getattr(self.core, "schema", {}) or {}
                mis = desc.get("mismatch") or {}
                missing_ct = (len(mis.get("missing") or []) if isinstance(mis.get("missing"), list) else int(
                    mis.get("missing") or 0))
                unexpected_ct = (len(mis.get("unexpected") or []) if isinstance(mis.get("unexpected"), list) else int(
                    mis.get("unexpected") or 0))
                self.log(f"🧠 Model schema: layers={desc.get('layers')} heads={desc.get('heads')} "
                         f"hidden={desc.get('hidden')} max_len={desc.get('max_len')} "
                         f"mismatch: missing={missing_ct} unexpected={unexpected_ct}", tag='sys')

                if (desc.get('missing', 0) or desc.get('unexpected', 0)):
                    self.log("⚠️ Checkpoint/schema mismatch detected — quality may be degraded. "
                             "Prefer exporting a checkpoint that exactly matches this architecture or provide a correct .meta.json.",
                             tag='sys')
            except FileNotFoundError:
                self.log("⚠️ Core could not auto-select a tokenizer; falling back to v8/v7/v6 search.", tag='sys')
                tok_fallback = self.resolve_tokenizer_path()
                if not tok_fallback:
                    self.log("❌ No tokenizer found in fallback resolver.", tag='sys'); return False
                self.core = get_global_core(model_path=path, tokenizer_path=tok_fallback, device='cpu', enable_retrieval=True, encoder_ckpt=None, max_len=getattr(self, 'max_len', 300), force_reload=True)

            self.current_model = os.path.basename(path)

            try:
                Vt = self.core.tokenizer.get_vocab_size()
                Ve = self._model_vocab_size(self.core.model)
                chosen_tok_path = getattr(self.core, "tokenizer_path", "(unknown)")
                chosen_tok_name = os.path.basename(chosen_tok_path) if chosen_tok_path != "(unknown)" else chosen_tok_path
                self.log(f"🧩 Tokenizer: {chosen_tok_path}  | vocab={Vt}  embed={Ve}", tag='sys')
                self.update_tokenizer_dropdown()
                if chosen_tok_name in getattr(self, "_tok_label_to_path", {}):
                    self.tokenizer_var.set(chosen_tok_name)
                else:
                    self.tokenizer_var.set("Auto (match model vocab)")
                if Ve is not None and Vt != Ve:
                    raise RuntimeError(f"Vocab mismatch: tokenizer={Vt}, embed={Ve}")
            except Exception as e:
                self.log(f"❌ Tokenizer/weights mismatch: {e}", tag='sys'); return False

            self.log(f"[{self.ts()}] 🔁 Loaded model: {self.current_model}", tag='sys')
            self.update_model_dropdown()
            return True

        except Exception as e:
            self.log(f"⚠️ Load error: {e}\n{tb.format_exc()}", tag='sys'); return False

    def watch_new_model(self):
        last_mtime=0
        while True:
            latest=self.find_latest_model()
            if latest:
                mt=os.path.getmtime(latest)
                if mt>last_mtime:
                    tok_path = self.get_selected_tokenizer_path()
                    ok=self.load_model(latest, tok_override=tok_path); last_mtime=mt
                    self.ui_log(f"[{self.ts()}] 🌙 REM/training finished — auto-switched to newest brain." if ok
                                else f"[{self.ts()}] 🌙 New model detected but failed to load.", tag='sys')
            time.sleep(5)

    # ----- intents / generation -----
    def on_send(self, _=None):
        self.stop_slow_type()
        text = self.input.get('1.0', 'end-1c').strip()
        self.input.delete('1.0', 'end')
        self._autosize_input()
        if not text:
            return

        self.log(f"[{self.ts()}] You > {text}", tag='ardor')

        # update emotion and HUD from the user's message
        self.update_emotion_from_text(text)

        if self.pending_plan and text.lower() in ("y", "yes", "ok", "confirm", "do it"):
            self.execute_plan(self.pending_plan);
            self.pending_plan = None;
            return
        it = self.parser.parse(text)
        if it: self.route_intents(it); return
        Thread(target=self.generate, args=(text,), daemon=True).start()

    def route_intents(self, intent: Intent):
        cfg = ALLOW[self.autonomy.get()]
        plan = []
        for act in intent.actions:
            t = act['type']
            if t in cfg['auto']: plan.append(("auto", act))
            elif t in cfg['confirm']: plan.append(("confirm", act))
            else: plan.append(("block", act))
        for mode, act in plan:
            self.log(f"• {mode.upper()}: {act}", tag='sys')
        if any(m == "confirm" for m, _ in plan):
            self.pending_plan = plan
            self.log("Type 'yes' to execute plan or continue chatting to ignore.", tag='sys'); return
        self.execute_plan(plan)

    def execute_plan(self, plan):
        for _,act in plan:
            t=act['type']
            if t=='set':
                k,v=act['key'],act['value']
                if k in DECODE_CFG:
                    DECODE_CFG[k]=v; self.log(f"🔧 {k} → {v}", tag='sys')
                else:
                    self.log(f"⚠️ Unknown param {k}", tag='sys')
            elif t=='switch_model':
                self.load_model(os.path.join(MODELS_DIR, act['name']), tok_override=self.get_selected_tokenizer_path())
            elif t=='rem':
                delay=int(act.get('delay',0))
                if delay>0:
                    self.log(f"⏳ REM will start in {delay}s…", tag='sys')
                    Timer(delay, lambda: Thread(target=self.run_rem, daemon=True).start()).start()
                else:
                    Thread(target=self.run_rem, daemon=True).start()
            elif t=='train':
                Thread(target=self.run_train, daemon=True).start()

    def run_rem(self):
        script = ROOT_DIR / "Cerebrum" / "CorticalIntegration" / "REM.py"
        self.launch_process([sys.executable, str(script.resolve())], "sleeping")

    def run_train(self):
        script = ROOT_DIR / "Cerebrum" / "Cortex" / "neural_plasticity_training.py"
        self.launch_process([sys.executable, str(script.resolve())], "learning")

    def launch_process(self, cmd, label):
        kwargs={"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP, "close_fds": True} if WIN else {"start_new_session": True}
        self.anim_event.clear(); Thread(target=self.animate_state, args=(label,), daemon=True).start()
        try:
            self.proc=subprocess.Popen(cmd, **kwargs); self.ui_log(f"↗ Started: {' '.join(cmd)}", tag='sys')
            Thread(target=self.monitor_process, daemon=True).start()
        except Exception as e:
            self.ui_log(f"⚠️ Launch error: {e}", tag='sys')

    def monitor_process(self):
        rc=self.proc.wait(); self.anim_event.set()
        if rc==0:
            self.ui_log("✅ Process finished — switching to newest model…", tag='sys')
            latest=self.find_latest_model()
            if latest: self.ui(self.load_model, latest, self.get_selected_tokenizer_path())
        else:
            self.ui_log("⏰ Process interrupted or failed.", tag='sys')
        self.proc=None

    def _is_junky(self, s: str) -> bool:
        s = re.sub(r'\s+', ' ', s).strip()
        if not s:
            return True
        letters = sum(ch.isalpha() for ch in s)
        if letters == 0:
            return True
        if len(s) < 30:
            punct = sum(ch in '.,;:!?()[]{}\'"\\/|_~`^+-=*' for ch in s)
            if letters < 6 or punct > letters * 1.8:
                return True
            return False
        punct = sum(ch in '.,;:!?()[]{}\'"\\/|_~`^+-=*' for ch in s)
        if punct > letters * 1.4:
            return True
        if any(len(tok) > 50 for tok in s.split()):
            return True
        return False

    def animate_state(self, label):
        i=0
        while not self.anim_event.is_set():
            bar="█"*(i%12)+"-"*(11-(i%12)); i+=1
            self.ui_log(f"\n   [{bar}]  {label}\n", tag='sys'); time.sleep(0.25)

    def _cleanup_wiki_noise(self, text: str) -> str:
        text = re.sub(r'(?:\r?\n)*(References|External links|See also)\b.*', '', text, flags=re.I | re.S)
        lines = []
        for ln in text.splitlines():
            if re.search(r'\b(living people|births|deaths|Category:|alumni|footballers)\b', ln, flags=re.I):
                if len(ln.split()) < 6: continue
            lines.append(ln)
        text = "\n".join(lines)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def generate(self, prompt: str):
        if not self.core:
            self.ui_log("⚠️ Model not loaded.", tag='sys');
            return

        self.ui(self.show_thinking)

        try:
            temp = DECODE_CFG['temperature']
            topp = DECODE_CFG['top_p']
            rep = DECODE_CFG['rep_penalty']

            if getattr(self, "aeternum", None) and getattr(self, "last_emotion", None):
                try:
                    nm = getattr(self.aeternum, "neuromodulators", None)
                    if nm is not None:
                        scales = nm.compute_decoding_scales(self.last_emotion)
                        temp *= scales.get("temperature_scale", 1.0)
                        topp *= scales.get("top_p_scale", 1.0)
                        rep *= scales.get("rep_penalty_scale", 1.0)
                except Exception as e:
                    self.ui_log(f"[Aeternum] neuromodulator error: {e}", tag='sys')

            temp = DECODE_CFG['temperature']
            topp = DECODE_CFG['top_p']
            rep = DECODE_CFG['rep_penalty']

            if getattr(self, "aeternum", None) and getattr(self, "last_emotion", None):
                try:
                    nm = getattr(self.aeternum, "neuromodulators", None)
                    if nm is not None:
                        scales = nm.compute_decoding_scales(self.last_emotion)
                        temp *= scales.get("temperature_scale", 1.0)
                        topp *= scales.get("top_p_scale", 1.0)
                        rep *= scales.get("rep_penalty_scale", 1.0)
                except Exception as e:
                    self.ui_log(f"[Aeternum] neuromodulator error: {e}", tag='sys')

            raw = self.core.generate_text(
                prompt,
                temperature=temp,
                top_p=topp,
                rep_penalty=rep,
                ngram_block=DECODE_CFG['ngram'],
                persona_primer=""
            )

            ans = self._cleanup_wiki_noise(raw or "")
            txt = (ans if ans else raw).strip()

            if not txt or self._is_junky(txt):
                raw2 = self.core.generate_text(
                    prompt,
                    temperature=0.8,
                    top_p=0.92,
                    top_k=0,
                    typical_p=0.30,
                    rep_penalty=1.05,
                    ngram_block=0,
                    suppress_vague=False,
                    persona_primer="",
                    stop_on_eos=False
                )
                txt2 = self._cleanup_wiki_noise(raw2 or "").strip()
                if txt2 and (not self._is_junky(txt2)) and (len(txt2) >= max(len(txt) + 10, 30)):
                    self.ui_log("↻ Fallback decode (safer settings)…", tag='sys')
                    txt = txt2

            self.ui(self.replace_thinking_with_stream, txt, 'ardor')

        except Exception as e:
            self.ui_log(f"⚠️ {e}", tag='sys')

    # ----- REM status -----
    def simulate_rem_sleep(self):
        path=str((ARTIFACTS_REM_DIR / "rem_status.json").resolve())
        while True:
            if os.path.exists(path):
                try:
                    d=json.load(open(path))
                    pct=d.get("progress",0); epoch=d.get("epoch",0); tot=d.get("total_epochs",1); loss=d.get("loss",0.0)
                    ts=time.strftime('%H:%M:%S', time.localtime(d.get('timestamp', time.time())))
                    self.ui(self.status.configure, text=f"REM {epoch}/{tot} loss={loss}  {pct}% @ {ts}")
                    if not self.progress.winfo_ismapped():
                        self.ui(self.progress.pack, {'pady':6})
                        self.ui(self.status.pack)
                    self.ui(setattr, self.progress, 'value', pct)
                except Exception:
                    self.ui(self.status.configure, text="REM: status error")
            else:
                if self.progress.winfo_ismapped():
                    self.ui(self.progress.pack_forget)
                    self.ui(self.status.pack_forget)
            time.sleep(2)

    # ----- misc -----
    def ts(self): return time.strftime('%H:%M:%S')

    def run_tests(self):
        self.log("\n[Tests] Running intent parser tests…\n", tag='sys')
        p=self.parser
        cases=[("enter REM in 2 minutes",("rem",)),
               ("enter REM at 02:00",("rem",)),
               ("train now",("train",)),
               ("set temperature to 0.8 and repetition penalty 1.25; top-p 0.9",("set","set","set")),
               ("switch model to Ardor_VIII.pt",("switch_model",)),
               ("ngram 5 and sleep at 11:30 pm",("set","rem"))]
        ok=0
        for s,kinds in cases:
            it=p.parse(s); got=tuple(a['type'] for a in it.actions); good=all(k in got for k in kinds)
            if 'rem' in kinds and any(a['type']=='rem' for a in it.actions):
                rem=[a for a in it.actions if a['type']=='rem'][0]; good=good and (rem.get('delay',0)>0)
            ok+=int(good); self.log(f"Passed {ok}/{len(cases)} tests.\n", tag='sys')

    # ----- handoff: switch to Atelier (Qt) -----
    def _find_atelier_script(self) -> Path | None:
        cands = [
            ROOT_DIR / "Hephaestus" / "Atelier.py",
            ROOT_DIR / "Atelier.py",
            Path(__file__).resolve().parent / "Hephaestus" / "Atelier.py",
        ]
        for p in cands:
            if p.exists():
                return p
        return None

    def switch_to_atelier(self):
        script = self._find_atelier_script()
        if not script:
            self.log("❌ Could not find Hephaestus/Atelier.py", tag='sys')
            return
        try:
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP, "close_fds": True} if WIN else {"start_new_session": True}
            subprocess.Popen([sys.executable, str(script), str(ROOT_DIR)], **kwargs)
            self.after(50, self.destroy)
        except Exception as e:
            self.log(f"⚠️ Failed to launch Atelier: {e}", tag='sys')

# ----- boot -----
if __name__ == "__main__":
    print("[Ardor GUI] Booting…", flush=True)
    app = ArdorGUI()
    try:
        app.update_idletasks(); app.deiconify(); app.lift()
        app.attributes("-topmost", True); app.after(200, lambda: app.attributes("-topmost", False))
    except Exception as e:
        print(f"[Ardor GUI] deiconify/lift failed: {e}", flush=True)
    def _tk_excepthook(exc, val, tbk):
        import traceback as _tb
        print("\n[Ardor GUI] Tk callback exception:", file=sys.stderr, flush=True)
        _tb.print_exception(exc, val, tbk)
    app.report_callback_exception = _tk_excepthook
    try:
        style = ttk.Style(app); style.theme_use('clam')
    except Exception as e:
        print(f"[Ardor GUI] theme set failed: {e}", flush=True)
    print("[Ardor GUI] Entering mainloop…", flush=True)
    app.mainloop()
    print("[Ardor GUI] mainloop exited", flush=True)
