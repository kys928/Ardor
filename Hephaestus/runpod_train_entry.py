#!/usr/bin/env python3
"""Infrastructure adapter for the canonical Ardor prompt-generation trainer.

The canonical trainer keeps persistent data paths under /workspace/Ardor.
Inside the RunPod agent image, code lives under /opt/Ardor. This adapter only
redirects the trainer's code-import root; it does not alter training stages,
data formats, tokenizer paths, checkpoint formats, or optimization behavior.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_ROOT = REPO_ROOT / "Cerebrum" / "Cortex"

for path in (str(REPO_ROOT), str(CORTEX_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Cerebrum.Cortex import neural_plasticity_training as trainer


def main() -> None:
    trainer.ARDOR_ROOT = Path(
        os.environ.get("ARDOR_CODE_ROOT", str(REPO_ROOT))
    ).resolve()
    trainer.main()


if __name__ == "__main__":
    main()
