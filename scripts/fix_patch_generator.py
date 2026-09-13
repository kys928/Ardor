from pathlib import Path

path = Path("scripts/patch_conversation_audit.py")
text = path.read_text(encoding="utf-8")
old = 'Path("Erratum/ardor_v14a4_conversation_audit.py").write_text(audit, encoding="utf-8")'
new = 'audit = audit.replace(\'\\\\"\', \'"\')\nPath("Erratum/ardor_v14a4_conversation_audit.py").write_text(audit, encoding="utf-8")'
if old not in text:
    raise RuntimeError("audit write anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
