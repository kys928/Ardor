from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

try:
    from asciimatics.screen import Screen
except Exception as e:
    raise SystemExit(
        "This tryout needs asciimatics. Install with: pip install asciimatics\n"
        f"Import error: {e}"
    )

TITLE = "ARDOR"
FPS = 24
PROMPT_PREFIX = "> "
MAX_MSGS = 6

# Dense ramp from faint to solid.
RAMP = " .'`^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Toggle this if you want cyan later.
USE_COLOR = False


@dataclass
class ModeState:
    mode: str = "idle"  # idle | thinking | speaking | rem | error
    t0: float = 0.0

    def age(self) -> float:
        return time.time() - self.t0


class FaceField:
    def __init__(self) -> None:
        self.state = ModeState("idle", time.time())
        self.speak_until = 0.0
        self.pending_reply: Optional[str] = None
        self.reply_at = 0.0
        self.input_buffer = ""
        self.messages: List[str] = [
            "Ardor asciimatics face tryout.",
            "Commands: /quit /clear /mode idle|thinking|speaking|rem|error",
        ]

        self._last_ui_signature: Optional[str] = None
        self._last_w = 0
        self._last_h = 0

    def set_mode(self, mode: str) -> None:
        if mode != self.state.mode:
            self.state = ModeState(mode, time.time())

    def append(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.messages.append(line)
        self.messages = self.messages[-MAX_MSGS:]
        self._last_ui_signature = None

    def submit(self, prompt: str) -> bool:
        prompt = prompt.strip()
        if not prompt:
            return True

        self.append(PROMPT_PREFIX + prompt)

        if prompt == "/quit":
            return False

        if prompt == "/clear":
            self.messages = []
            self._last_ui_signature = None
            return True

        if prompt.startswith("/mode "):
            mode = prompt.split(" ", 1)[1].strip().lower()
            if mode in {"idle", "thinking", "speaking", "rem", "error"}:
                self.set_mode(mode)
                self.append(f"[mode] {mode}")
            else:
                self.append("[bad mode]")
            return True

        # demo fake reply cycle
        self.set_mode("thinking")
        self.pending_reply = f"ardor: {prompt[::-1][:64]}"
        self.reply_at = time.time() + 1.15
        return True

    def update(self) -> None:
        now = time.time()
        if self.pending_reply and now >= self.reply_at:
            self.append(self.pending_reply)
            self.pending_reply = None
            self.set_mode("speaking")
            self.speak_until = now + 1.25
        elif self.state.mode == "speaking" and now >= self.speak_until:
            self.set_mode("idle")

    def _face_params(self, t: float) -> Dict[str, float | str]:
        mode = self.state.mode
        age = self.state.age()

        blink = 1.0 if (math.sin(age * 0.88) + 0.18 * math.sin(age * 2.1)) > 1.10 else 0.0
        sway_x = 0.018 * math.sin(age * 0.41)
        sway_y = 0.012 * math.sin(age * 0.57)

        if mode == "thinking":
            eye_open = 0.42
            brow_drop = -0.070
            mouth_open = 0.015
            aura = 0.10
            jitter = 0.006
            asym = 0.010 * math.sin(age * 1.2)
            socket_depth = 1.30
        elif mode == "speaking":
            eye_open = 0.56
            brow_drop = -0.030
            mouth_open = 0.025 + 0.10 * ((math.sin(age * 11.5) + 1.0) * 0.5)
            aura = 0.06
            jitter = 0.004
            asym = 0.006 * math.sin(age * 2.8)
            socket_depth = 1.10
        elif mode == "rem":
            eye_open = 0.10
            brow_drop = 0.010
            mouth_open = 0.010
            aura = 0.16
            jitter = 0.008
            asym = 0.010 * math.sin(age * 0.6)
            socket_depth = 0.85
        elif mode == "error":
            eye_open = 0.34
            brow_drop = -0.09
            mouth_open = 0.05
            aura = 0.12
            jitter = 0.016
            asym = 0.028 * math.sin(age * 6.0)
            socket_depth = 1.45
        else:
            eye_open = 0.52
            brow_drop = -0.025
            mouth_open = 0.012
            aura = 0.05
            jitter = 0.003
            asym = 0.004 * math.sin(age * 0.8)
            socket_depth = 1.00

        if blink > 0.5:
            eye_open *= 0.05

        gaze_x = 0.020 * math.sin(age * 0.6) + (0.010 if mode == "thinking" else 0.0)
        gaze_y = 0.010 * math.sin(age * 0.45)

        return {
            "mode": mode,
            "t": t,
            "blink": blink,
            "sway_x": sway_x,
            "sway_y": sway_y,
            "eye_open": eye_open,
            "brow_drop": brow_drop,
            "mouth_open": mouth_open,
            "aura": aura,
            "jitter": jitter,
            "asym": asym,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "socket_depth": socket_depth,
        }

    @staticmethod
    def _ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float) -> float:
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2

    @staticmethod
    def _gauss(x: float, y: float, cx: float, cy: float, rx: float, ry: float, k: float = 1.0) -> float:
        return math.exp(-k * (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))

    def _sample_intensity(self, x: float, y: float, p: Dict[str, float | str]) -> float:
        mode = str(p["mode"])
        eye_open = float(p["eye_open"])
        brow_drop = float(p["brow_drop"])
        mouth_open = float(p["mouth_open"])
        asym = float(p["asym"])
        socket_depth = float(p["socket_depth"])

        # Overall head mass
        head = 1.10 * self._gauss(x, y, 0.0, -0.02, 0.56, 0.82, 2.4)

        # Skull / forehead plate
        forehead = 0.42 * self._gauss(x, y, 0.0, -0.60, 0.22, 0.13, 7.0)
        crown = 0.18 * self._gauss(x, y, 0.0, -0.82, 0.12, 0.06, 14.0)

        # Cheekbones
        cheek_l = 0.34 * self._gauss(x, y, -0.27, 0.12, 0.16, 0.20, 9.0)
        cheek_r = 0.34 * self._gauss(x, y, 0.27, 0.12, 0.16, 0.20, 9.0)

        # Jaw and chin
        jaw_l = 0.20 * self._gauss(x, y, -0.24, 0.43, 0.13, 0.19, 10.0)
        jaw_r = 0.20 * self._gauss(x, y, 0.24, 0.43, 0.13, 0.19, 10.0)
        chin = 0.24 * self._gauss(x, y, 0.0, 0.66, 0.12, 0.08, 18.0)

        val = head + forehead + crown + cheek_l + cheek_r + jaw_l + jaw_r + chin

        # Hollow temples
        temples = 0.16 * self._gauss(x, y, -0.42, -0.26, 0.10, 0.18, 13.0)
        temples += 0.16 * self._gauss(x, y, 0.42, -0.26, 0.10, 0.18, 13.0)
        val -= temples

        # Eye sockets
        lx = -0.20 + float(p["gaze_x"]) + asym
        rx = 0.20 + float(p["gaze_x"]) - asym
        ey = -0.10 + float(p["gaze_y"])

        l_socket = self._gauss(x, y, lx, ey, 0.16, 0.09 * max(0.25, eye_open), 14.0)
        r_socket = self._gauss(x, y, rx, ey, 0.16, 0.09 * max(0.25, eye_open), 14.0)
        val -= socket_depth * (l_socket + r_socket)

        # Dark inner eye pits to get closer to the reference
        l_inner = self._gauss(x, y, lx, ey + 0.01, 0.10, 0.055 * max(0.25, eye_open), 20.0)
        r_inner = self._gauss(x, y, rx, ey + 0.01, 0.10, 0.055 * max(0.25, eye_open), 20.0)
        val -= 0.75 * (l_inner + r_inner)

        # Brow shelf
        l_brow = self._gauss(x, y, -0.20 + asym, -0.21 + brow_drop, 0.17, 0.030, 40.0)
        r_brow = self._gauss(x, y, 0.20 - asym, -0.21 + brow_drop * 0.7, 0.17, 0.030, 40.0)
        val += 0.23 * (l_brow + r_brow)

        # Nose bridge
        bridge = self._gauss(x, y, 0.0, 0.00, 0.035, 0.23, 26.0)
        tip = self._gauss(x, y, 0.0, 0.20, 0.075, 0.055, 42.0)
        nostril_l = self._gauss(x, y, -0.050, 0.23, 0.020, 0.015, 60.0)
        nostril_r = self._gauss(x, y, 0.050, 0.23, 0.020, 0.015, 60.0)
        val += 0.48 * bridge + 0.22 * tip
        val -= 0.10 * (nostril_l + nostril_r)

        # Philtrum + upper lip line
        philtrum = self._gauss(x, y, 0.0, 0.33, 0.020, 0.05, 70.0)
        upper_lip = self._gauss(x, y, 0.0, 0.42, 0.14, 0.018, 90.0)
        mouth_hole = self._gauss(x, y, 0.0, 0.46, 0.090, 0.018 + mouth_open, 70.0)
        lower_lip_shadow = self._gauss(x, y, 0.0, 0.50, 0.11, 0.020, 70.0)

        val += 0.10 * philtrum
        val += 0.18 * upper_lip
        val -= 0.32 * mouth_hole
        val -= 0.05 * lower_lip_shadow

        # Face edge fade — avoid hard cartoon outline
        edge = self._ellipse(x, y, 0.0, -0.02, 0.60, 0.88)
        val *= max(0.0, 1.10 - 0.18 * edge)

        # Slight asymmetry makes it feel less fake
        val += 0.018 * math.sin(7.0 * x + 4.0 * y + float(p["t"]) * 0.7)

        # Aura / digital haze
        aura = float(p["aura"]) * self._gauss(x, y, 0.0, -0.03, 0.90, 1.10, 3.4)
        val += aura

        if mode == "error":
            val += 0.08 * math.sin(12.0 * x + 8.0 * y + float(p["t"]) * 4.0)
        elif mode == "rem":
            val += 0.04 * math.sin(8.0 * x - 5.0 * y + float(p["t"]) * 1.6)

        return val

    def _ui_signature(self, w: int, h: int) -> str:
        return repr((w, h, self.state.mode, tuple(self.messages), self.input_buffer))

    def _draw_ui(self, screen: Screen, w: int, h: int) -> None:
        ui_sig = self._ui_signature(w, h)
        if ui_sig == self._last_ui_signature and w == self._last_w and h == self._last_h:
            return

        msg_top = max(0, h - (MAX_MSGS + 3))

        # Clear UI zone only
        for y in range(msg_top, h):
            screen.print_at(" " * max(1, w - 1), 0, y)

        # Divider
        if msg_top - 1 >= 0:
            screen.print_at("-" * max(1, w - 1), 0, msg_top - 1)

        y = msg_top
        for msg in self.messages[-MAX_MSGS:]:
            screen.print_at(msg[: max(1, w - 1)], 0, y)
            y += 1

        status = f"[{self.state.mode}]"
        screen.print_at(status[: max(1, w - 1)], 0, h - 2)
        screen.print_at((PROMPT_PREFIX + self.input_buffer)[: max(1, w - 1)], 0, h - 1)

        self._last_ui_signature = ui_sig
        self._last_w = w
        self._last_h = h

    def _draw_face(self, screen: Screen, w: int, h: int) -> None:
        msg_top = max(0, h - (MAX_MSGS + 3))
        top = 2
        bottom = max(top + 8, msg_top - 2)

        # Clear animation area only
        blank = " " * max(1, w - 1)
        for y in range(top, bottom):
            screen.print_at(blank, 0, y)

        title_x = max(0, (w - len(TITLE)) // 2)
        screen.print_at(TITLE, title_x, 0)

        p = self._face_params(time.time())

        cx = w // 2
        cy = top + (bottom - top) // 2 - 1

        face_w = max(24, min(90, int(w * 0.34)))
        face_h = max(14, min(42, int((bottom - top) * 0.46)))

        for sy in range(top, bottom):
            yn = ((sy - cy) / max(1, face_h)) * 2.15 + float(p["sway_y"])

            for sx in range(max(0, cx - face_w), min(w - 1, cx + face_w)):
                xn = ((sx - cx) / max(1, face_w)) * 1.55 + float(p["sway_x"])
                xn2 = xn + float(p["jitter"]) * math.sin(yn * 6.2 + float(p["t"]) * 1.7)
                yn2 = yn + float(p["jitter"]) * math.cos(xn * 7.4 - float(p["t"]) * 1.3)

                v = self._sample_intensity(xn2, yn2, p)

                # Vignette
                vignette = 1.0 - 0.22 * min(1.0, (abs(xn2) ** 1.7 + abs(yn2) ** 1.7) / 2.0)
                v *= vignette

                # Sparse point-cloud face with better structure
                seed = 0.5 + 0.5 * math.sin(sx * 11.7 + sy * 78.1 + float(p["t"]) * 1.5)
                threshold = 0.19 + 0.07 * seed

                if v > threshold:
                    density = max(0.0, min(1.0, (v - threshold) / 0.92))
                    idx = int(density * (len(RAMP) - 1))
                    ch = RAMP[idx]
                    screen.print_at(ch, sx, sy)
                else:
                    # very faint ambient field around the head
                    halo = math.exp(-3.7 * (((xn2) / 0.95) ** 2 + ((yn2 + 0.02) / 1.10) ** 2))
                    if halo > 0.22:
                        mist_seed = math.sin(sx * 2.7 + sy * 1.9 + float(p["t"]) * (0.55 if p["mode"] == "rem" else 0.95))
                        if mist_seed > 0.985:
                            screen.print_at("." if p["mode"] != "rem" else ":", sx, sy)

    def draw(self, screen: Screen) -> None:
        self.update()
        h, w = screen.height, screen.width
        self._draw_face(screen, w, h)
        self._draw_ui(screen, w, h)
        screen.refresh()


def app(screen: Screen) -> None:
    scene = FaceField()
    frame_dt = 1.0 / FPS
    running = True

    while running:
        start = time.time()
        scene.draw(screen)

        ev = screen.get_key()
        changed_input = False

        while ev is not None:
            if ev in (3, 17):  # Ctrl-C / Ctrl-Q
                return
            if ev in (10, 13):  # Enter
                running = scene.submit(scene.input_buffer)
                scene.input_buffer = ""
                changed_input = True
            elif ev in (8, 127, Screen.KEY_BACK):
                scene.input_buffer = scene.input_buffer[:-1]
                changed_input = True
            elif ev == Screen.KEY_ESCAPE:
                return
            elif 32 <= ev <= 126:
                scene.input_buffer += chr(ev)
                changed_input = True
            ev = screen.get_key()

        if changed_input:
            scene._last_ui_signature = None

        elapsed = time.time() - start
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)


def main() -> None:
    Screen.wrapper(app)


if __name__ == "__main__":
    main()