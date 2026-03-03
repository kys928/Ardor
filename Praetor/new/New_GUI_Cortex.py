# GUI_Cortex.py — Ardor HUD (modern, EXE-aware, feature-parity with Cyan Edition)

from __future__ import annotations

# ── Hi-DPI awareness (Windows) ────────────────────────────────────────
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os, time, json, math, subprocess, platform, re, importlib, inspect, traceback as tb, shutil, signal
from threading import Thread, Event, Timer
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ── “conversational not Wikipedia” primer (kept for parity) ───────────
STYLE_PRIME = (
    "You are Ardor. Answer in clear, natural, conversational English. "
    "Do not write like Wikipedia. Avoid categories, section headers like "
    "'References', 'External links', or 'See also', and boilerplate such as "
    "'living people', 'births', 'deaths', or 'Category:'. "
    "Give a direct, helpful answer; be concise unless asked for detail."
)

# ── Paths & discovery (EXE can live in ArdorGUI\; project stays in C:\Davidko) ─
def base_dir() -> Path:
    """Folder the app runs from (EXE: _MEIPASS/dist; PY: script folder)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()

def read_ardor_home_hint() -> Path | None:
    """Optional text file next to EXE with absolute path to project root."""
    hint = base_dir() / "ardor_home.txt"
    try:
        if hint.exists():
            p = Path(hint.read_text(encoding="utf-8").strip()).expanduser().resolve()
            return p if p.exists() else None
    except Exception:
        pass
    return None

def auto_find_project_root() -> Path:
    env = os.environ.get("ARDOR_HOME")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    hint = read_ardor_home_hint()
    if hint:
        return hint
    markers = {"tokenizer_v6.json", "broca_decoder.py"}
    cur = base_dir()
    for _ in range(5):
        if any((cur / m).exists() for m in markers) or any(cur.glob("*.pt")):
            return cur
        cur = cur.parent
    parent = base_dir().parent
    if parent.exists():
        return parent
    return base_dir()

APP_DIR  = base_dir()
ROOT_DIR = auto_find_project_root()
os.chdir(ROOT_DIR)  # scripts run with project root cwd

def resource_path(rel: str | Path, *, root: Path = ROOT_DIR) -> Path:
    return (root / rel).resolve()

# ── ArdorCore import resolution ───────────────────────────────────────
sys.path.append(str(resource_path("Cerebrum/Cortex")))
def resolve_ArdorCore():
    for modname, path in [("prefrontal_cortex","Cerebrum/Cortex"),
                          ("prefrontal_cortex_autonomy_v2",".")]:
        try:
            abspath = str(resource_path(path))
            if abspath and abspath not in sys.path:
                sys.path.append(abspath)
            mod = importlib.import_module(modname)
            cls = getattr(mod, "ArdorCore", None)
            if inspect.isclass(cls):
                return cls
        except Exception:
            pass
    return None
ArdorCoreCls = resolve_ArdorCore()

# ── Intents (same semantics as Cyan Edition) ─────────────────────────
class Intent:
    def __init__(self): self.actions=[]
    def add(self,t,**k): self.actions.append({"type":t,**k}); return self
    def __bool__(self): return bool(self.actions)

class IntentParser:
    re_rem   = re.compile(r"\b(rem|sleep)\b",re.I)
    re_train = re.compile(r"\btrain(?!ing)\b",re.I)
    re_temp  = re.compile(r"\btemp(?:erature)?\s*(?:to|=)?\s*([0-9]*\.?[0-9]+)",re.I)
    re_topp  = re.compile(r"\btop[-_ ]?p\s*(?:to|=)?\s*(0?\.\d+|1(?:\.0)?)",re.I)
    re_rep   = re.compile(r"\brep(?:etition)?(?:\s*pen(?:alty)?)?\s*(?:to|=)?\s*([0-9]*\.?[0-9]+)",re.I)
    re_ngram = re.compile(r"\bngram\b.*?(\d+)",re.I)
    re_model = re.compile(r"\b(?:model|switch(?:\s*to)?)\s+([\w_.-]+\.pt)\b",re.I)
    re_delay = re.compile(r"\bin\s+(\d+)\s*(s|sec|seconds|m|min|minutes|h|hours)\b",re.I)
    re_abstime = re.compile(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",re.I)
    @staticmethod
    def _to_seconds(n,u): u=u.lower(); return int(n)*(1 if u.startswith('s') else 60 if u.startswith('m') else 3600)
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
        m = self.re_temp.search(text);  it.add("set", key="temperature", value=float(m.group(1))) if m else None
        m = self.re_topp.search(text);  it.add("set", key="top_p", value=float(m.group(1)))      if m else None
        m = self.re_rep.search(text);   it.add("set", key="rep_pen", value=float(m.group(1)))    if m else None
        m = self.re_ngram.search(text); it.add("set", key="ngram", value=int(m.group(1)))        if m else None
        m = self.re_model.search(text); it.add("switch_model", name=m.group(1))                  if m else None
        return it

AUTONOMY_LEVELS=("Off","Assist","Copilot","Autonomous")
ALLOW={
 "Off":        {"auto": set(),                           "confirm":{"set","switch_model","rem","train"}},
 "Assist":     {"auto":{"set","switch_model"},           "confirm":{"rem","train"}},
 "Copilot":    {"auto":{"set","switch_model","rem"},     "confirm":{"train"}},
 "Autonomous": {"auto":{"set","switch_model","rem"},     "confirm":{"train"}},
}

# ── Theme ────────────────────────────────────────────────────────────
DARK  = {'bg':'#0e0f12','fg':'#e8eaed','panel':'#121317','stroke':'#1f2127','accent':'#19c8ea','accent2':'#7be3ff','muted':'#9aa0a6','input':'#17181d'}
LIGHT = {'bg':'#ffffff','fg':'#0b0d12','panel':'#f5f6f7','stroke':'#e5e7eb','accent':'#0aa8c6','accent2':'#62cfe6','muted':'#5f6368','input':'#f1f3f4'}

# ── Minimal, responsive HUD ──────────────────────────────────────────
class ModernHUD(tk.Canvas):
    def __init__(self, master, theme, **kw):
        super().__init__(master, highlightthickness=0, bd=0, **kw)
        self.theme=theme
        self.configure(bg=theme['panel'])
        self._t=0.0
        self.bind("<Configure>", self._on_resize)
        self.after(33, self._tick)
    def set_theme(self, theme):
        self.theme=theme
        self.configure(bg=theme['panel'])
        self._draw(animate=False)
    def _on_resize(self, _): self._draw(animate=False)
    def _tick(self):
        self._t += 0.033
        self._draw(animate=True)
        self.after(33, self._tick)
    def _draw(self, animate: bool):
        w=max(self.winfo_width(), 10); h=max(self.winfo_height(), 10)
        cx, cy = w//2, h//2
        r = max(18, int(min(w, h) * 0.22))
        self.delete("hud")
        self.create_oval(cx-r-8, cy-r-8, cx+r+8, cy+r+8, outline=self.theme['stroke'], width=2, tags="hud")
        self.create_oval(cx-r, cy-r, cx+r, cy+r, outline=self.theme['accent'], width=3, tags="hud")
        a = (self._t*1.5) % (2*math.pi)
        px = cx + (r-3)*math.cos(a); py = cy + (r-3)*math.sin(a)
        self.create_oval(px-3, py-3, px+3, py+3, fill=self.theme['accent2'], outline="", tags="hud")

# ── App ──────────────────────────────────────────────────────────────
DECODE_CFG={"temperature":0.60,"top_p":0.85,"rep_pen":1.60,"ngram":4}
WIN=platform.system()=="Windows"

def find_system_python() -> str | None:
    for candidate in ("py", "python3.11", "python3", "python"):
        p = shutil.which(candidate)
        if p: return p
    return None

class ArdorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        # Tk scaling
        try:
            ppi = self.winfo_fpixels('1i')
            self.tk.call('tk', 'scaling', max(1.0, ppi/72.0))
        except Exception: pass

        self.title("Ardor")
        try:
            icon_p = (APP_DIR / "assets/ardor.ico")
            if icon_p.exists(): self.iconbitmap(default=str(icon_p))
        except Exception: pass

        sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
        W=min(1200,int(sw*0.9)); H=min(820,int(sh*0.9))
        self.geometry(f"{W}x{H}"); self.minsize(960,640)

        self.theme=DARK; self.parser=IntentParser(); self.core=None
        self.current_model=None
        self.anim_event=Event()
        self.pending_plan=None; self.autonomy=tk.StringVar(value="Assist")
        self.child_procs: list[subprocess.Popen] = []
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_style(); self._build_ui(W,H); self.set_theme(self.theme)

        self.log(f"Root: {ROOT_DIR}")
        latest=self.find_latest_model()
        if latest: self.load_model(latest)
        else: self.log("No models (*.pt) found in root or models folder.")

        Thread(target=self.watch_new_model, daemon=True).start()
        self.bind('<Return>', self._on_return); self.bind('<Shift-Return>', lambda e: None)
        self.bind('<F1>', lambda e: self.run_tests())   # parity: F1 tests

    # UI + style
    def _build_style(self):
        self.style=ttk.Style(self)
        try: self.style.theme_use('clam')
        except Exception: pass
        self._apply_theme(self.theme)

    def _apply_theme(self, t):
        s=self.style
        s.configure("TFrame", background=t['bg'])
        s.configure("Panel.TFrame", background=t['panel'])
        s.configure("TLabel", background=t['bg'], foreground=t['fg'])
        s.configure("Muted.TLabel", background=t['bg'], foreground=t['muted'])
        s.configure("TButton", background=t['panel'], foreground=t['fg'], borderwidth=0, padding=(12,7))
        s.map("TButton", background=[("active", t['accent'])], foreground=[("active", "#000")])
        s.configure("Accent.TButton", background=t['accent'], foreground="#000", padding=(12,7))
        s.map("Accent.TButton", background=[("active", t['accent2'])])
        s.configure("TCombobox", fieldbackground=t['panel'], background=t['panel'], foreground=t['fg'])
        s.configure("Horizontal.TProgressbar", troughcolor=t['panel'], background=t['accent'])
        self.configure(bg=t['bg'])

    def _build_ui(self, W, H):
        top = ttk.Frame(self, style="Panel.TFrame"); top.pack(fill=tk.X)
        row = ttk.Frame(top, style="Panel.TFrame"); row.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(row, text="Ardor", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(row, text="Autonomy:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8,4))
        self.autonomy_box=ttk.Combobox(row, values=AUTONOMY_LEVELS, textvariable=self.autonomy, state="readonly", width=12)
        self.autonomy_box.pack(side=tk.LEFT)

        right = ttk.Frame(row, style="Panel.TFrame"); right.pack(side=tk.RIGHT)
        ttk.Button(right, text="Dark",  command=lambda:self.set_theme(DARK),  style="TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(right, text="Light", command=lambda:self.set_theme(LIGHT), style="TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(right, text="Train", command=self._start_train, style="TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(right, text="REM",   command=self._start_rem,   style="TButton").pack(side=tk.LEFT, padx=6)

        center = ttk.Frame(self); center.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10,10))

        hud_frame = ttk.Frame(center, style="Panel.TFrame"); hud_frame.pack(fill=tk.X)
        self.hud = ModernHUD(hud_frame, theme=self.theme, height=120)
        self.hud.pack(fill=tk.X, expand=True, padx=8, pady=8)

        term_frame = ttk.Frame(center); term_frame.pack(fill=tk.BOTH, expand=True, pady=(8,8))
        self.term = tk.Text(term_frame, wrap='word', undo=False, relief=tk.FLAT,
                            bg=self.theme['bg'], fg=self.theme['fg'], insertbackground=self.theme['accent'],
                            font=("Consolas", 11))
        self.term.configure(spacing1=3, spacing3=4, padx=6, pady=6)
        self.term.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        term_scroll = ttk.Scrollbar(term_frame, orient=tk.VERTICAL, command=self.term.yview)
        self.term['yscrollcommand'] = term_scroll.set
        term_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._enable_mousewheel(self.term)  # parity: smooth wheel/trackpad

        input_frame = ttk.Frame(center, style="Panel.TFrame"); input_frame.pack(fill=tk.X, pady=(4,0))
        self.input = tk.Text(input_frame, height=3, wrap='word', relief=tk.FLAT,
                             bg=self.theme['input'], fg=self.theme['fg'], insertbackground=self.theme['accent'],
                             font=("Consolas", 11))
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,8), pady=10)
        self.input.bind('<KeyRelease>', self._autosize_input)  # parity: auto-size
        ttk.Button(input_frame, text="Send", command=self.on_send, style="Accent.TButton").pack(side=tk.RIGHT, padx=10, pady=10)

        status = ttk.Frame(self, style="Panel.TFrame"); status.pack(fill=tk.X)
        self.status_lbl = ttk.Label(status, text="Ready", style="Muted.TLabel")
        self.status_lbl.pack(side=tk.LEFT, padx=12, pady=6)
        self.progress = ttk.Progressbar(status, mode='determinate', length=220, style="Horizontal.TProgressbar")
        self.progress.pack(side=tk.RIGHT, padx=12, pady=6)
        self._status_bar = status  # for show/hide

        Thread(target=self._watch_rem_status, daemon=True).start()

    # UX helpers (parity)
    def _enable_mousewheel(self, widget: tk.Text):
        widget.bind("<MouseWheel>", lambda e: widget.yview_scroll(-1*(e.delta//120), "units"))
        widget.bind("<Button-4>",  lambda e: widget.yview_scroll(-1, "units"))
        widget.bind("<Button-5>",  lambda e: widget.yview_scroll(+1, "units"))

    def _autosize_input(self, _=None):
        try:
            lines=int(self.input.index('end-1c').split('.')[0])
            want=min(max(lines, 3), 7)
            if want!=int(self.input.cget('height')):
                self.input.configure(height=want)
        except Exception:
            pass

    def set_theme(self, theme):
        self.theme = theme
        self._apply_theme(theme)
        self.term.configure(bg=theme['bg'], fg=theme['fg'], insertbackground=theme['accent'])
        self.input.configure(bg=theme['input'], fg=theme['fg'], insertbackground=theme['accent'])
        self.hud.set_theme(theme)

    # Keys
    def _on_return(self, e):
        if (e.state & 0x0001):  # Shift
            return
        self.on_send(); return 'break'

    # Model management (search root and optional models dir for parity)
    def _candidate_model_paths(self):
        paths = []
        for d in [ROOT_DIR, ROOT_DIR / "Cerebrum/Models/Ardor"]:
            if d.exists():
                paths.extend(d.glob("*.pt"))
        return paths

    def find_latest_model(self):
        pts=self._candidate_model_paths()
        return max(pts, key=lambda p: p.stat().st_mtime) if pts else None

    def resolve_tokenizer_path(self):
        candidates = [
            ROOT_DIR / "tokenizer_v6.json",
            ROOT_DIR / "tokenizer_v5.json",
            ROOT_DIR / "Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v6.json",
            ROOT_DIR / "ProjectTokenizer/ardor_tokenizer/tokenizer_v6.json",
            ROOT_DIR / "Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v5.json",
            ROOT_DIR / "ProjectTokenizer/ardor_tokenizer/tokenizer_v5.json",
            Path(r"C:/Davidko/tokenizer_v6.json"),
            Path(r"C:/Davidko/tokenizer_v5.json"),
        ]
        for p in candidates:
            if p.exists(): return str(p)
        return None

    def load_model(self, path: Path) -> bool:
        try:
            tok=self.resolve_tokenizer_path()
            if not tok:
                self.log("Tokenizer not found (v6/v5)."); return False
            if not callable(ArdorCoreCls):
                self.log("ArdorCore class not found."); return False
            try:
                origin=inspect.getsourcefile(ArdorCoreCls) or str(ArdorCoreCls)
                self.log(f"Core: {origin}")
            except Exception: pass
            self.core=ArdorCoreCls(model_path=str(path), tokenizer_path=tok, device='cpu')
            self.current_model=path.name
            self.status_lbl.config(text=f"Loaded {self.current_model}")
            self.log(f"Loaded model: {self.current_model}")
            return True
        except Exception as e:
            self.log(f"Load error: {e}\n{tb.format_exc()}"); return False

    def watch_new_model(self):
        last_mtime=0
        while True:
            latest=self.find_latest_model()
            if latest:
                mt=latest.stat().st_mtime
                if mt>last_mtime:
                    ok=self.load_model(latest); last_mtime=mt
                    self.log("Auto-switched to newest model." if ok else "Detected new model but failed to load.")
            time.sleep(5)

    # Intents
    def on_send(self, _=None):
        text=self.input.get('1.0','end-1c').strip()
        self.input.delete('1.0','end')
        if not text: return

        # parity: confirm plan with “yes”
        if self.pending_plan and text.lower() in ("y","yes","ok","confirm","do it"):
            self.execute_plan(self.pending_plan); self.pending_plan=None
            return

        self.log(f"You ▸ {text}")
        it=IntentParser().parse(text)
        if it: return self.route_intents(it)
        Thread(target=self.generate, args=(text,), daemon=True).start()

    def route_intents(self, intent:Intent):
        cfg=ALLOW[self.autonomy.get()]; plan=[]
        for act in intent.actions:
            t=act['type']
            if t in cfg['auto']: plan.append(("auto",act))
            elif t in cfg['confirm']: plan.append(("confirm",act))
            else: plan.append(("block",act))
        for mode,act in plan:
            self.log(f"• {mode.upper()}: {act}")
        if any(m=="confirm" for m,_ in plan):
            self.pending_plan=plan; self.log("Type 'yes' to execute plan or continue chatting to ignore.")
            return
        self.execute_plan(plan)

    def execute_plan(self, plan):
        for _,act in plan:
            t=act['type']
            if t=='set':
                k,v=act['key'],act['value']
                if k in DECODE_CFG: DECODE_CFG[k]=v; self.log(f"Set {k} → {v}")
                else: self.log(f"Unknown param {k}")
            elif t=='switch_model':
                # search both root and models dir
                target = None
                for d in [ROOT_DIR, ROOT_DIR / "Cerebrum/Models/Ardor"]:
                    p = d / act['name']
                    if p.exists(): target = p; break
                if target: self.load_model(target)
                else: self.log(f"Model not found: {act['name']}")
            elif t=='rem':
                delay=int(act.get('delay',0))
                if delay>0:
                    self.log(f"REM in {delay}s…")
                    Timer(delay, lambda: Thread(target=self.run_rem, daemon=True).start()).start()
                else:
                    Thread(target=self.run_rem, daemon=True).start()
            elif t=='train':
                Thread(target=self.run_train, daemon=True).start()

    def _start_rem(self):  Thread(target=self.run_rem, daemon=True).start()
    def _start_train(self): Thread(target=self.run_train, daemon=True).start()

    # Process launchers (use ROOT_DIR as CWD so scripts see the same tree)
    def run_rem(self):
        script = ROOT_DIR / "REM.py"
        if not script.exists():
            self.log(f"Missing REM script: {script}"); return
        py = find_system_python()
        if not py: self.log("No system Python found in PATH."); return
        self.launch_process([py, str(script)], "REM running")

    def run_train(self):
        script = ROOT_DIR / "neural_plasticity_training.py"
        if not script.exists():
            self.log(f"Missing training script: {script}"); return
        py = find_system_python()
        if not py: self.log("No system Python found in PATH."); return
        self.launch_process([py, str(script)], "Training")

    def launch_process(self, cmd, label):
        kwargs={}
        if WIN: kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.anim_event.clear(); Thread(target=self._animate_state, args=(label,), daemon=True).start()
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), **kwargs)
            if not hasattr(self, "child_procs"): self.child_procs=[]
            self.child_procs.append(proc)
            self.log(f"Started: {' '.join(cmd)} (cwd={ROOT_DIR})")
            Thread(target=self._monitor_process, args=(proc,), daemon=True).start()
        except Exception as e:
            self.log(f"Launch error: {e}")

    def _monitor_process(self, proc: subprocess.Popen):
        proc.wait()
        self.anim_event.set()
        try: self.child_procs.remove(proc)
        except ValueError: pass
        if proc.returncode == 0:
            self.log("Process finished — checking newest model…")
            latest=self.find_latest_model()
            if latest: self.load_model(latest)
        else:
            self.log("Process interrupted or failed.")

    def _animate_state(self, label):
        i=0
        while not self.anim_event.is_set():
            dots="."*(i%4); i+=1
            self.status_lbl.config(text=f"{label}{dots}")
            time.sleep(0.3)
        self.status_lbl.config(text="Ready")

    # Safe exit: confirm + stop REM/training
    def on_close(self):
        running = [p for p in getattr(self, "child_procs", []) if p.poll() is None]
        if running:
            if not messagebox.askyesno("Confirm Exit", "Training/REM is running.\nStop all tasks and exit?"):
                return
            self._terminate_children(timeout=7.0)
        self.destroy()

    def _terminate_children(self, timeout: float = 5.0):
        for p in list(getattr(self, "child_procs", [])):
            if p.poll() is None:
                try:
                    if WIN: p.send_signal(signal.CTRL_BREAK_EVENT)
                    else:   p.terminate()
                except Exception: pass
        end=time.time()+timeout
        while time.time()<end and any(p.poll() is None for p in getattr(self, "child_procs", [])):
            time.sleep(0.1)
        for p in list(getattr(self, "child_procs", [])):
            if p.poll() is None:
                try: p.kill()
                except Exception: pass

    # Generation
    def generate(self, prompt:str):
        if not self.core: self.log("Model not loaded."); return
        self.log("Ardor ▸ (thinking)")
        try:
            ans=self.core.generate_text(prompt, temperature=DECODE_CFG['temperature'],
                                        top_p=DECODE_CFG['top_p'], rep_pen=DECODE_CFG['rep_pen'],
                                        ngram_block=DECODE_CFG['ngram'])
            self.log(ans.strip()+"\n")
        except Exception as e:
            self.log(f"{e}")

    # REM status (parity: show/hide bar based on file presence)
    def _watch_rem_status(self):
        path = ROOT_DIR / "rem_status.json"
        bar_visible = True
        def hide_bar():
            nonlocal bar_visible
            if bar_visible:
                self.progress.pack_forget()
                bar_visible=False
        def show_bar():
            nonlocal bar_visible
            if not bar_visible:
                self.progress.pack(side=tk.RIGHT, padx=12, pady=6)
                bar_visible=True
        while True:
            try:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as fp:
                        d=json.load(fp)
                    pct=int(d.get("progress",0)); epoch=int(d.get("epoch",0)); tot=int(d.get("total_epochs",1)); loss=float(d.get("loss",0.0))
                    ts=time.strftime('%H:%M:%S', time.localtime(d.get('timestamp', time.time())))
                    self.progress['value']=pct; show_bar()
                    self.status_lbl.config(text=f"REM {epoch}/{tot}  loss {loss:.4f}  {pct}% @ {ts}")
                else:
                    hide_bar()
                    self.status_lbl.config(text="Ready")
                time.sleep(2)
            except Exception:
                time.sleep(2)

    # Logging + tests
    def log(self, msg:str):
        self.term.insert(tk.END, msg + "\n"); self.term.see(tk.END)

    def ts(self): return time.strftime('%H:%M:%S')

    def run_tests(self):
        self.log("\n[Tests] Running intent parser tests…\n")
        p=IntentParser()
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
            ok+=int(good); self.log(f"{s!r} => {got} :: {'OK' if good else 'FAIL'}\n")
        self.log(f"Passed {ok}/{len(cases)} tests.\n")

# ── Boot ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    app=ArdorGUI()
    app.mainloop()
