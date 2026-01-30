from __future__ import annotations
import csv, time
from .protocols import EmotionState

class AeternumTelemetry:
    def __init__(self, path="aeternum_telemetry.csv"):
        self.path = path
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t","valence","arousal","dominance","surprise","uncertainty","anxiety","alertness","reward","safe","stance","plan"])

    def log(self, st: EmotionState):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([time.time(), st.valence, st.arousal, st.dominance, st.surprise,
                        st.uncertainty, st.anxiety, st.alertness, st.reward_tone,
                        int(st.safe_mode), st.stance, st.plan])
