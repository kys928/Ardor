#!/usr/bin/env python3
"""Control-plane entrypoint that registers current fixed-purpose experimental runners."""
from __future__ import annotations

import runpod_control

runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_format_ab_u100")
runpod_control.FIXED_PURPOSE_RUNNERS.add("v14a4_signal_quality_audit")


if __name__ == "__main__":
    runpod_control.main()
