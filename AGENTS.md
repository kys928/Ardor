# Ardor agent instructions

## Project intent
Ardor is a neurobiologically themed AI system. Preserve neurobiological naming unless explicitly told to rename or flatten.

## Canonical top-level ownership
- Cerebrum/: core model architecture, training, inference, cognition
- Aeternum/: memory, replay, consolidation, affective/state-related systems
- Praetor/: GUI, app shell, operator-facing interfaces
- Erratum/: diagnostics, repair scripts, migrations, one-off maintenance tools
- Hephaestus/: build/tooling/infrastructure-related code where applicable

## Hard rules
- Do not rename core neurobiological modules without explicit instruction.
- Do not create duplicate loaders, duplicate tokenizers, or parallel inference entrypoints unless explicitly requested.
- Prefer patching existing canonical files over adding new variants like *_fixed2.py or *_new.py.
- Preserve checkpoint compatibility unless the task explicitly requests an architecture migration.
- If changing training or inference code, explain the impact on checkpoints, tokenizer compatibility, and memory interfaces.

## Safety checks before proposing changes
- Read relevant files fully before editing.
- Search for duplicate implementations before adding a new one.
- Prefer small, reviewable diffs.
- Summarize touched files and architectural consequences.

## Style
- Keep explanations in two layers:
  1. plain software-engineering explanation
  2. neurobiological analogy if relevant
