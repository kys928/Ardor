# GUI_Cortex.py — Ardor Autonomy HUD (Elodin Retro Orange style + scanline + grid)
from __future__ import annotations
import os, sys, time, json, math, random, subprocess, platform, re, importlib, inspect, traceback as tb
from threading import Thread, Event, Timer
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from typing import Optional
from pathlib import Path
from collections import deque
import sys
from pathlib import Path


# Project root = .../ProjectArdor
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aeternum.AeternumCore import AeternumCore, AeternumConfig

from Cerebrum.Cortex.prefrontal_cortex import get_global_core

# ── Aeternum Singleton ───────────────────────────────────────────────
_AETERNUM_SINGLETON = None





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

# ---------------- core import ----------------
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
sys.path.append("../Cerebrum/Cortex")

def _import_module_by_path(py_path: Path):
    name = py_path.stem + "_dyn"
    loader = SourceFileLoader(name, str(py_path))
    spec = spec_from_loader(name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def resolve_ArdorCore():
    mod_name = os.environ.get("ARDOR_CORE_MODULE", "") or ""
    cls_name = os.environ.get("ARDOR_CORE_CLASS", "ArdorCore") or "ArdorCore"
    mod_name, cls_name = mod_name.strip(), cls_name.strip() or "ArdorCore"

    cortex_dir = (ROOT_DIR / "Cerebrum" / "Cortex")
    if cortex_dir.exists():
        p = str(cortex_dir.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    if mod_name:
        try:
            if mod_name.lower().endswith(".py") and Path(mod_name).is_file():
                mod = _import_module_by_path(Path(mod_name))
            else:
                mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if inspect.isclass(cls): return cls
        except Exception:
            pass

    for candidate in ("prefrontal_cortex",):
        try:
            mod = importlib.import_module(candidate)
            cls = getattr(mod, "ArdorCore", None)
            if inspect.isclass(cls): return cls
        except Exception:
            pass

    try:
        if cortex_dir.exists():
            for py in cortex_dir.glob("*.py"):
                try:
                    txt = py.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if re.search(r"\bclass\s+ArdorCore\b", txt):
                    try:
                        mod = _import_module_by_path(py)
                        cls = getattr(mod, "ArdorCore", None)
                        if inspect.isclass(cls): return cls
                    except Exception:
                        continue
    except Exception:
        pass

    return None

ArdorCoreCls = resolve_ArdorCore()

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

# ---------------- Eye HUD (Lined / Iridescent Orange Iris) ----------------
class HUD(tk.Canvas):
    """
    Moving orange iris with concentric tracks + radial ticks, rotating reticles, bloom, and natural blinks.
    Keeps the same public methods as the previous HUD so *no other GUI logic changes*.
    """

    ORANGE = dict(
        ring="#ff9100",
        tick="#ff7a1a",
        track="#ffb354",
        glow="#ffb347",
        spec="#fff2e0",
        highlight="#ffffff",
    )

    def __init__(self, master, theme, **kw):
        super().__init__(master, highlightthickness=0, **kw)
        self.theme = theme or RETRO_DARK
        self.configure(bg=self.theme['bg'])

        # geometry (initialized on first draw)
        self.cx = 0
        self.cy = 0
        self.rx = 180
        self.ry = 110
        self.r_iris = 62
        self.r_pupil = 24
        self.max_offset = 0.38

        # state
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.mouse_local = None
        self.bias_local = None
        self.bias_until = 0.0

        # blinking
        self.lid_open = 1.0
        self.blinking = False
        self.reopening = False
        self.next_blink = 10**18

        # micro jitter + rotation
        self.micro_t = random.random() * 100.0
        self.t = 0.0
        self.rot = 0.0

        # display-driven tint from emotion (optional)
        self._valence = 0.0
        self._arousal = 0.0

        # items
        self.items = {}
        self.bloom_layers = []
        self.track_ids = []      # concentric rings
        self.tick_ids = []       # radial ticks (rebuilt each frame)
        self.reticle_layers = [] # segmented arcs

        self.bind("<Configure>", lambda _e: self._redraw())
        self._redraw()
        self.after(30, self._tick)

    # ---- compatibility shims
    def set_theme(self, theme):
        self.theme = theme or RETRO_DARK
        self.configure(bg=self.theme.get('bg', '#000000'))
        self._redraw()

    def look_at_root(self, x_root: int, y_root: int):
        cxr, cyr = self.winfo_rootx(), self.winfo_rooty()
        self.mouse_local = (x_root - cxr, y_root - cyr)

    def bias_towards_widget(self, widget: tk.Widget, _duration: float = 1.5):
        # Keep param name `_duration` because current GUI calls it that way.
        try:
            widget.update_idletasks()
            wx, wy = widget.winfo_rootx(), widget.winfo_rooty()
            ww, wh = max(widget.winfo_width(), 1), max(widget.winfo_height(), 1)
            cxr, cyr = self.winfo_rootx(), self.winfo_rooty()
            self.bias_local = (wx + ww / 2 - cxr, wy + wh / 2 - cyr)
            self.bias_until = time.time() + float(_duration)
        except Exception:
            pass

    def set_status(self, _status: str, _revert_after: float | None = None):
        # API compatibility; we don't need status coloring here.
        if _revert_after:
            self.after(int(float(_revert_after) * 1000), lambda: None)

    def set_from_emotion_state(self, st):
        """
        Optional: use emotion to slightly modulate bloom intensity / motion.
        We keep the iris orange; we just make it breathe more when arousal is high.
        """
        try:
            self._valence = float(getattr(st, "valence", 0.0))
            self._arousal = float(getattr(st, "arousal", 0.0))
        except Exception:
            self._valence, self._arousal = 0.0, 0.0

    # ---- internal helpers
    def _mix_to_theme(self, c: str, t: float) -> str:
        bg = self.theme.get('bg', '#000000')
        # nudge towards bg for integration
        return _mix(c, bg, max(0.0, min(1.0, t)))

    def _limit_offset(self, dx, dy):
        mx = self.rx * self.max_offset
        my = self.ry * self.max_offset
        if dx == 0 and dy == 0:
            return 0.0, 0.0
        ux, uy = dx / mx, dy / my
        mag = math.hypot(ux, uy)
        if mag <= 1.0:
            return dx, dy
        ux /= mag
        uy /= mag
        return ux * mx, uy * my

    def _blend_target(self):
        now = time.time()
        bias_w = 0.78 if (now < self.bias_until and self.bias_local) else 0.0
        default = (self.cx, self.cy + self.ry * 0.35)
        mouse = self.mouse_local or default
        if self.bias_local:
            tx = (1 - bias_w) * mouse[0] + bias_w * self.bias_local[0]
            ty = (1 - bias_w) * mouse[1] + bias_w * self.bias_local[1]
            return (tx, ty)
        return mouse

    def _max_radius_inside_ellipse(self, ix, iy, samples=24):
        dx, dy = ix - self.cx, iy - self.cy
        rx2, ry2 = self.rx * self.rx, self.ry * self.ry
        if (dx * dx) / rx2 + (dy * dy) / ry2 >= 1.0:
            return 0.0
        tmin = float('inf')
        for k in range(samples):
            th = (2 * math.pi) * k / samples
            ux, uy = math.cos(th), math.sin(th)
            A = (ux * ux) / rx2 + (uy * uy) / ry2
            B = 2 * (dx * ux / rx2 + dy * uy / ry2)
            C = (dx * dx) / rx2 + (dy * dy) / ry2 - 1.0
            disc = B * B - 4 * A * C
            if disc <= 0:
                continue
            t = (-B + math.sqrt(disc)) / (2 * A)
            if t > 0 and t < tmin:
                tmin = t
        return max(0.0, tmin if tmin != float('inf') else 0.0)

    def _make_reticles(self, center, color):
        cx, cy = center
        self.reticle_layers.clear()

        def ring(R, segs, span_deg, speed):
            arcs = []
            for i in range(segs):
                start = i * (360 / segs)
                aid = self.create_arc(cx - R, cy - R, cx + R, cy + R,
                                      start=start, extent=span_deg,
                                      style='arc', outline=color, width=2)
                arcs.append(aid)
            self.reticle_layers.append({'arcs': arcs, 'offset': 0.0, 'speed': speed, 'span': 360 / segs})

        ring(self.r_iris + 14, 48, 6, +0.6)
        ring(self.r_iris + 26, 36, 7, -0.45)

    def _update_eyelids(self):
        now = time.time()
        if not self.blinking and now >= self.next_blink:
            self.blinking = True
            self.reopening = False
        if self.blinking:
            if not self.reopening:
                self.lid_open = max(0.0, self.lid_open - 0.20)
                if self.lid_open <= 0.0:
                    self.reopening = True
            else:
                self.lid_open = min(1.0, self.lid_open + 0.14)
                if self.lid_open >= 1.0:
                    self.blinking = False
                    self.next_blink = now + random.uniform(2.6, 5.5)

        open_y = self.ry * self.lid_open
        pad = 2
        self.coords(self.items['top_lid'], self.cx - self.rx - pad, self.cy - self.ry - pad,
                    self.cx + self.rx + pad, self.cy - open_y)
        self.coords(self.items['bot_lid'], self.cx - self.rx - pad, self.cy + open_y,
                    self.cx + self.rx + pad, self.cy + self.ry + pad)

    def _redraw(self):
        # full rebuild on resize/theme
        self.delete('all')
        self.items = {}
        self.bloom_layers = []
        self.track_ids = []
        self.tick_ids = []
        self.reticle_layers = []

        w = self.winfo_width() or int(float(self.cget('width') or 0)) or 1
        h = self.winfo_height() or int(float(self.cget('height') or 0)) or 1

        self.cx, self.cy = w // 2, h // 2

        # scale the eye with viewport
        s = min(w, h)
        self.rx = max(140, int(s * 0.24))
        self.ry = max(84,  int(s * 0.15))
        self.r_iris = max(46, int(s * 0.085))
        self.r_pupil = max(18, int(self.r_iris * 0.38))

        pal = self.ORANGE
        ring_c = pal['ring']
        track_c = pal['track']
        tick_c = pal['tick']
        spec_c = pal['spec']
        high_c = pal['highlight']
        glow_c = pal['glow']

        # subtle grid (keeps your retro vibe)
        grid_c = self.theme.get('grid', '#2f2216')
        step = 28
        for x in range(0, w, step):
            self.create_line(x, 0, x, h, fill=grid_c)
        for y in range(0, h, step):
            self.create_line(0, y, w, y, fill=grid_c)

        sclera_fill = self._mix_to_theme("#0f0e0c", 0.15) if self.theme.get('bg') == RETRO_DARK.get('bg') else self._mix_to_theme("#fffaf2", 0.10)
        sclera_outline = self._mix_to_theme(ring_c, 0.55 if self.theme.get('bg') == RETRO_DARK.get('bg') else 0.25)

        self.items['sclera'] = self.create_oval(self.cx - self.rx, self.cy - self.ry,
                                                self.cx + self.rx, self.cy + self.ry,
                                                fill=sclera_fill, outline=sclera_outline, width=2)

        # pupil
        self.items['pupil'] = self.create_oval(self.cx - self.r_pupil, self.cy - self.r_pupil,
                                               self.cx + self.r_pupil, self.cy + self.r_pupil,
                                               fill='#000000', outline='')

        # iris boundary
        self.items['iris_border'] = self.create_oval(self.cx - self.r_iris, self.cy - self.r_iris,
                                                     self.cx + self.r_iris, self.cy + self.r_iris,
                                                     outline=ring_c, width=2)

        # concentric tracks
        inner = self.r_pupil + 4
        outer = self.r_iris - 2
        rings = 8
        for j in range(rings):
            r = inner + (j + 1) * (outer - inner) / (rings + 1)
            oid = self.create_oval(self.cx - r, self.cy - r, self.cx + r, self.cy + r,
                                   outline=self._mix_to_theme(track_c, 0.18 + 0.08 * j), width=1)
            self.track_ids.append(oid)

        # initial radial ticks (rebuilt every frame, but we draw once here too)
        def band(r0, r1, N, thickness_step=5):
            for k in range(N):
                th = math.radians((360 / N) * k)
                x0 = self.cx + r0 * math.cos(th)
                y0 = self.cy + r0 * math.sin(th)
                x1 = self.cx + r1 * math.cos(th)
                y1 = self.cy + r1 * math.sin(th)
                lw = 1 if k % thickness_step else 2
                lid = self.create_line(x0, y0, x1, y1, fill=tick_c, width=lw, capstyle=tk.ROUND)
                self.tick_ids.append(lid)

        band(self.r_pupil + 4, self.r_pupil + 10, 64)
        band(self.r_pupil + 14, self.r_pupil + 22, 48)
        band(self.r_pupil + 26, self.r_iris - 6, 36)

        # reticles
        self._make_reticles(center=(self.cx, self.cy), color=ring_c)

        # specular arc + highlight
        Rspec = self.r_iris + 7
        self.items['spec_ring'] = self.create_arc(self.cx - Rspec, self.cy - Rspec, self.cx + Rspec, self.cy + Rspec,
                                                  start=25, extent=54, style='arc', outline=spec_c, width=3)
        self.items['highlight'] = self.create_oval(self.cx - 8, self.cy - 8, self.cx - 2, self.cy - 2,
                                                   fill=high_c, outline='')

        # bloom halos
        for i in range(6):
            r = self.r_iris + 10 + i * 6
            oid = self.create_oval(self.cx - r, self.cy - r, self.cx + r, self.cy + r,
                                   outline=self._mix_to_theme(glow_c, 0.55 + 0.35 * (i / 6)), width=2)
            self.bloom_layers.append(oid)

        # lids (masks)
        pad = 2
        bg = self.theme.get('bg', '#000000')
        self.items['top_lid'] = self.create_rectangle(self.cx - self.rx - pad, self.cy - self.ry - pad,
                                                      self.cx + self.rx + pad, self.cy - self.ry,
                                                      fill=bg, outline=bg)
        self.items['bot_lid'] = self.create_rectangle(self.cx - self.rx - pad, self.cy + self.ry,
                                                      self.cx + self.rx + pad, self.cy + self.ry + pad,
                                                      fill=bg, outline=bg)

    def _tick(self):
        pal = self.ORANGE
        ring_c = pal['ring']
        tick_c = pal['tick']
        spec_c = pal['spec']
        high_c = pal['highlight']
        glow_c = pal['glow']

        self.t += 0.03
        # speed up a touch with arousal (feels alive)
        self.rot = (self.rot + (0.8 + 0.6 * max(0.0, min(1.0, self._arousal)))) % 360

        tx, ty = self._blend_target()

        # micro drift
        self.micro_t += 0.03
        tx += math.sin(self.micro_t * 1.7) * 2.2
        ty += math.cos(self.micro_t * 1.3) * 1.6

        dx, dy = self._limit_offset(tx - self.cx, ty - self.cy)

        a = 0.18
        self.gaze_x = (1 - a) * self.gaze_x + a * dx
        self.gaze_y = (1 - a) * self.gaze_y + a * dy
        ix, iy = self.cx + self.gaze_x, self.cy + self.gaze_y

        # pupil + iris move
        self.coords(self.items['pupil'], ix - self.r_pupil, iy - self.r_pupil, ix + self.r_pupil, iy + self.r_pupil)
        self.coords(self.items['iris_border'], ix - self.r_iris, iy - self.r_iris, ix + self.r_iris, iy + self.r_iris)

        # tracks
        inner = self.r_pupil + 4
        outer = self.r_iris - 2
        for j, oid in enumerate(self.track_ids):
            r = inner + (j + 1) * (outer - inner) / (len(self.track_ids) + 1)
            self.coords(oid, ix - r, iy - r, ix + r, iy + r)

        # rebuild radial ticks with rotation
        for lid in self.tick_ids:
            self.delete(lid)
        self.tick_ids.clear()

        bands = [
            (self.r_pupil + 4, self.r_pupil + 10, 64, +0.8),
            (self.r_pupil + 14, self.r_pupil + 22, 48, -0.6),
            (self.r_pupil + 26, self.r_iris - 6, 36, +0.4),
        ]
        for r0, r1, N, spd in bands:
            for k in range(N):
                ang = math.radians((360 / N) * k + self.rot * spd)
                x0 = ix + r0 * math.cos(ang)
                y0 = iy + r0 * math.sin(ang)
                x1 = ix + r1 * math.cos(ang)
                y1 = iy + r1 * math.sin(ang)
                lw = 1 if k % 5 else 2
                lid = self.create_line(x0, y0, x1, y1, fill=tick_c, width=lw, capstyle=tk.ROUND)
                self.tick_ids.append(lid)

        # reticles rotate + clamp inside sclera
        Rmax = max(0.0, self._max_radius_inside_ellipse(ix, iy) - 6)
        for li, layer in enumerate(self.reticle_layers):
            base = self.r_iris + (14 if li == 0 else 26)
            R = min(base, 0.88 * Rmax)
            for aid in layer['arcs']:
                self.coords(aid, ix - R, iy - R, ix + R, iy + R)
            layer['offset'] = (layer['offset'] + layer['speed']) % 360
            for i, aid in enumerate(layer['arcs']):
                self.itemconfigure(aid, start=(i * layer['span'] + layer['offset']) % 360, outline=ring_c)

        # specular + highlight
        Rspec = min(self.r_iris + 7, max(0.0, Rmax - 4))
        self.coords(self.items['spec_ring'], ix - Rspec, iy - Rspec, ix + Rspec, iy + Rspec)
        self.itemconfigure(self.items['spec_ring'], outline=spec_c)

        hx, hy = ix - self.r_pupil * 0.6, iy - self.r_pupil * 0.6
        self.coords(self.items['highlight'], hx - 6, hy - 6, hx, hy)
        self.itemconfigure(self.items['highlight'], fill=high_c)

        # bloom halos (breathe more with arousal)
        breath = 0.5 + 0.5 * math.sin(self.t * (1.2 + 0.9 * self._arousal))
        for i, oid in enumerate(self.bloom_layers):
            r0 = self.r_iris + 10 + i * 6
            self.coords(oid, ix - r0, iy - r0, ix + r0, iy + r0)
            self.itemconfigure(
                oid,
                outline=self._mix_to_theme(glow_c, 0.55 + 0.35 * (i / max(1, len(self.bloom_layers))) * (0.8 + 0.2 * breath)),
                width=1 + int(2 * breath),
            )

        self._update_eyelids()
        self.after(30, self._tick)


# ---------------- GUI ----------------
DECODE_CFG = {"temperature": 0.70, "top_p": 0.90, "rep_penalty": 1.15, "ngram": 4}

WIN_MODELS_PATH = Path(r"C:\Users\adm\PycharmProjects\ProjectArdor\Cerebrum\Models\Ardor")
if WIN_MODELS_PATH.exists():
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
        self.theme = RETRO_DARK
        self.parser = IntentParser()
        self.core = None

        # --- GUI-side chat state (prevents GUI from “helpfully” recreating core) ---
        self._core_model_path = None
        self._core_tokenizer_path = None

        # --- Optional controls ---
        self.show_facts_var = tk.BooleanVar(value=False)
        self.chat_history_len_var = tk.IntVar(value=10)  # 0 = off (send only current user message)
        self.store_good_only_var = tk.BooleanVar(value=True)

        # --- GUI chat history buffer (only used if chat_history_len_var > 0) ---
        self.chat_history = deque(maxlen=200)  # hard cap; effective length controlled by chat_history_len_var


        # Aeternum is owned by PFC singleton (bridge). GUI is display-only.
        self.aeternum = None
        self.last_emotion = None


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

        # --- Optional UI controls (command-driven)
        # UI elements removed by design; toggle via chat commands:
        #   - type: facts_panel
        #   - type: store_good_only

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

        # Output (chat + optional facts panel)
        self.trans_frame = tk.Frame(self.panes, bg=self.theme['bg'])

        self.out_panes = tk.PanedWindow(self.trans_frame, orient=tk.HORIZONTAL, sashwidth=6,
                                        bg=self.theme['bg'], bd=0, relief=tk.FLAT)
        self.out_panes.pack(fill=tk.BOTH, expand=True, padx=12)

        # Left: chat terminal
        self.term_frame = tk.Frame(self.out_panes, bg=self.theme['bg'])
        self.term = tk.Text(self.term_frame, bg=self.theme['bg'], fg=self.theme['text'],
                            insertbackground=self.theme['accent3'], font=("Consolas", 11),
                            height=16, wrap='word', relief=tk.FLAT, bd=1,
                            highlightthickness=1, highlightbackground=self.theme['stroke'])
        self.term.pack(fill=tk.BOTH, expand=True)
        self._enable_mousewheel(self.term)
        self.out_panes.add(self.term_frame, stretch="always")

        # Right: facts panel (hidden by default)
        self.facts_frame = tk.Frame(self.out_panes, bg=self.theme['panel2'])
        self.facts = tk.Text(self.facts_frame, bg=self.theme['panel2'], fg=self.theme['text'],
                             font=("Consolas", 10), wrap='word', relief=tk.FLAT, bd=1,
                             highlightthickness=1, highlightbackground=self.theme['stroke'])
        self.facts.pack(fill=tk.BOTH, expand=True)
        # start hidden; toggle via checkbox
        self._facts_visible = False

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

    def _toggle_facts_panel(self):
        want = bool(self.show_facts_var.get())
        if want and not getattr(self, "_facts_visible", False):
            try:
                self.out_panes.add(self.facts_frame, minsize=160)
                self._facts_visible = True
            except Exception:
                pass
        elif (not want) and getattr(self, "_facts_visible", False):
            try:
                self.out_panes.forget(self.facts_frame)
                self._facts_visible = False
            except Exception:
                pass

    def _safe_set_env(self, key: str, val: str):
        try:
            os.environ[str(key)] = str(val)
        except Exception:
            pass

    def _call_generate_text_sigfiltered(self, prompt: str, **kwargs) -> str:
        """
        Calls core.generate_text() without guessing unsupported kwargs.
        This prevents runtime errors AND prevents devs from “fixing it” by reinitializing core.
        """
        fn = getattr(self.core, "generate_text", None)
        if not callable(fn):
            raise RuntimeError("core.generate_text() is not callable.")

        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            clean = {k: v for k, v in kwargs.items() if k in allowed}
            return fn(prompt, **clean)
        except Exception:
            # If inspect fails (rare), fall back to basic call.
            return fn(prompt)

    def _update_facts_panel_from_core(self):
        """
        Non-invasive: looks for common attributes; does nothing if not present.
        """
        if not getattr(self, "_facts_visible", False):
            return
        try:
            candidates = [
                getattr(self.core, "facts", None),
                getattr(self.core, "last_facts", None),
                getattr(self.core, "retrieval_facts", None),
                getattr(self.core, "last_retrieval", None),
            ]
            facts_obj = next((c for c in candidates if c), None)
            if facts_obj is None:
                return

            if isinstance(facts_obj, (list, tuple)):
                text = "\n".join(str(x) for x in facts_obj)
            elif isinstance(facts_obj, dict):
                text = "\n".join(f"{k}: {v}" for k, v in facts_obj.items())
            else:
                text = str(facts_obj)

            self.facts.delete("1.0", "end")
            self.facts.insert("end", text.strip() + "\n")
            self.facts.see("end")
        except Exception:
            return

    def _build_prompt_with_history(self, user_text: str) -> str:
        """
        GUI-side history is optional. If history_len_var == 0: send only the new user message.
        Otherwise: build a lightweight transcript.
        """
        keep = int(self.chat_history_len_var.get() or 0)
        if keep <= 0:
            return user_text

        # Append the new user turn into history first (as raw text lines)
        # We’ll also append assistant after generation.
        self.chat_history.append(("user", user_text))

        # Take last N turns and format
        turns = list(self.chat_history)[-keep * 2:]  # approximate: user+assistant pairs
        lines = []
        for role, txt in turns:
            if role == "user":
                lines.append(f"User: {txt}")
            else:
                lines.append(f"Assistant: {txt}")
        return "\n".join(lines).strip()

    def update_emotion_from_text(self, text: str):
        """Display-only: Aeternum updates happen inside PFC generate_text()."""
        try:
            if getattr(self, "core", None) is None:
                return
            aet = getattr(self.core, "aet", None)
            st = getattr(aet, "state", None) if aet is not None else None
            if st is not None:
                self.last_emotion = st
                self.hud.set_from_emotion_state(st)
        except Exception:
            return

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
        self.term.insert(tk.END, chunk)
        end = self.term.index('end-1c')
        start = f"{end} - {len(chunk)}c"
        try:
            self.term.tag_add(tag, start, end)
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
        try:
            pts=[f for f in os.listdir(MODELS_DIR) if f.lower().endswith('.pt')]
            pts.sort(key=lambda n: os.path.getmtime(os.path.join(MODELS_DIR, n)), reverse=True)
            return pts
        except Exception:
            return []

    def update_model_dropdown(self):
        models=self.list_models()
        self.model_box['values']=models
        want = self.current_model if self.current_model in models else (models[0] if models else "")
        self.model_var.set(want)

    def on_select_model(self):
        name = self.model_var.get().strip()
        if not name: return
        path=os.path.join(MODELS_DIR, name)
        if not os.path.isfile(path):
            self.log(f"⚠️ Not found: {path}", tag='sys'); return
        tok_path = self.get_selected_tokenizer_path()
        self.load_model(path, tok_override=tok_path)

    def find_latest_model(self):
        d=MODELS_DIR
        if not os.path.isdir(d): return None
        pts=[os.path.join(d,f) for f in os.listdir(d) if f.endswith('.pt')]
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

            if not callable(ArdorCoreCls):
                self.log("❌ ArdorCore class not found/callable.", tag='sys'); return False

            try:
                # Avoid reinitializing the singleton if we’re already on the same model+tokenizer.
                want_tok = (tok if tok and os.path.isfile(tok) else None)
                same_model = (self._core_model_path == os.path.abspath(path))
                same_tok = (self._core_tokenizer_path == (os.path.abspath(want_tok) if want_tok else None))

                self.core = get_global_core(
                    model_path=path,
                    tokenizer_path=want_tok,
                    device='cpu',
                    enable_retrieval=True,
                    force_reload=not (same_model and same_tok),
                )

                self._core_model_path = os.path.abspath(path)
                self._core_tokenizer_path = (os.path.abspath(want_tok) if want_tok else None)
                desc = self.core.model_schema() if hasattr(self.core, "model_schema") else getattr(self.core, "schema",
                                                                                                    {}) or {}
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

                self.core = get_global_core(
                    model_path=path,
                    tokenizer_path=tok if tok and os.path.isfile(tok) else None,
                    device="cpu",
                    enable_retrieval=False,
                    encoder_ckpt=r"..\Cerebrum\Models\Encoders\ArdorEncoder.pt",
                    max_len=300,
                    force_reload=True if (self._core_model_path != os.path.abspath(path)) else False,
                )

                self._core_model_path = os.path.abspath(path)
                self._core_tokenizer_path = (os.path.abspath(tok) if tok and os.path.isfile(tok) else None)

            self.current_model = os.path.basename(path)
            self.aeternum = getattr(self.core, "aet", None)
            if self.aeternum is not None:
                self.log("[Aeternum] Emotion core ready from prefrontal cortex.", tag="sys")
            else:
                self.log("[Aeternum] Emotion core unavailable (PFC init failed).", tag="sys")

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


        # Command-driven UI toggles (no other logic touched)
        cmd = text.strip().lower()
        if cmd == 'facts_panel':
            self.show_facts_var.set(not bool(self.show_facts_var.get()))
            self.ui(self._toggle_facts_panel)
            self.log(f"[ui] facts_panel -> {int(self.show_facts_var.get())}", tag='sys')
            return
        if cmd == 'store_good_only':
            self.store_good_only_var.set(not bool(self.store_good_only_var.get()))
            self.log(f"[ui] store_good_only -> {int(self.store_good_only_var.get())}", tag='sys')
            return


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
            self.ui_log("⚠️ Model not loaded.", tag='sys')
            return

        # Build prompt with optional GUI-side history (does NOT re-init core)
        built_prompt = self._build_prompt_with_history(prompt)

        self.ui(self.show_thinking)

        try:
            temp = DECODE_CFG['temperature']
            topp = DECODE_CFG['top_p']
            rep = DECODE_CFG['rep_penalty']

            # Memory write policy (env-driven; pairs with your PATCH 0 in PFC)
            # If "Store good only" is ON: default to NO writes during decode; allow PFC to decide / or write only if it deems good.
            if bool(self.store_good_only_var.get()):
                self._safe_set_env("ARDOR_NO_MEMORY_WRITES", "1")
                self._safe_set_env("ARDOR_STORE_GOOD_ONLY", "1")
            else:
                self._safe_set_env("ARDOR_NO_MEMORY_WRITES", "0")
                self._safe_set_env("ARDOR_STORE_GOOD_ONLY", "0")

            raw = self._call_generate_text_sigfiltered(
                built_prompt,
                temperature=temp,
                top_p=topp,
                rep_penalty=rep,
                ngram_block=DECODE_CFG['ngram']
            )

            ans = self._cleanup_wiki_noise(raw or "")
            txt = (ans if ans else raw).strip()

            # Optional fallback decode if junky (still does NOT re-init core)
            if not txt or self._is_junky(txt):
                raw2 = self._call_generate_text_sigfiltered(
                    built_prompt,
                    temperature=0.65,
                    top_p=0.90,
                    top_k=0,
                    typical_p=0.95,
                    rep_penalty=1.12,
                    ngram_block=4,
                    suppress_vague=False,
                    stop_on_eos=True
                )
                txt2 = self._cleanup_wiki_noise(raw2 or "").strip()
                if txt2 and (not self._is_junky(txt2)) and (len(txt2) >= max(len(txt) + 10, 30)):
                    self.ui_log("↻ Fallback decode (safer settings)…", tag='sys')
                    txt = txt2

            # If we are using GUI-side history, append assistant turn after final selection
            keep = int(self.chat_history_len_var.get() or 0)
            if keep > 0:
                self.chat_history.append(("assistant", txt))

            # Pull latest emotion state from PFC/Aeternum (display-only)
            try:
                aet = getattr(self.core, "aet", None)
                if aet is not None:
                    st = getattr(aet, "state", None)
                    if st is not None:
                        self.last_emotion = st
                        self.ui(self.hud.set_from_emotion_state, st)
            except Exception:
                pass

            # Facts panel update (best-effort; depends on what core exposes)
            self.ui(self._update_facts_panel_from_core)

            self.ui(self.replace_thinking_with_stream, txt, 'ardor')

        except Exception as e:
            self.ui_log(f"⚠️ {e}", tag='sys')

    # ----- REM status -----
    def simulate_rem_sleep(self):
        path="rem_status.json"
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
