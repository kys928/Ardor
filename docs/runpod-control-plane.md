# Ardor RunPod control plane

This control plane ports the proven GitHub -> RunPod -> Network Volume -> S3 pattern into Ardor without replacing Ardor's canonical trainer, tokenizer loader, checkpoint loader, or inference entrypoints.

## Boundary

GitHub Actions holds all RunPod API and S3 credentials. The GPU Pod receives only a validated job payload and the mounted Network Volume.

- image code: `/opt/Ardor`
- persistent volume mount: `/workspace`
- existing Ardor persistent tree: `/workspace/Ardor`
- control metadata: `/workspace/ardor-control`

`Hephaestus/runpod_train_entry.py` redirects only the canonical trainer's code-import root to `/opt/Ardor`. Data, tokenizer, run, and checkpoint paths remain under `/workspace/Ardor` exactly as the trainer expects.

## Required GitHub secrets

- `RUNPOD_API_KEY`
- `RUNPOD_S3_ACCESS_KEY_ID`
- `RUNPOD_S3_SECRET_ACCESS_KEY`

## Required GitHub variables

- `RUNPOD_NETWORK_VOLUME_ID`
- `RUNPOD_DATACENTER_ID`
- `RUNPOD_ALLOWED_GPU_TYPES`
- `RUNPOD_IMAGE_NAME`

Optional variables:

- `RUNPOD_S3_ENDPOINT` — derived as `https://s3api-<lowercase datacenter>.runpod.io/` when unset.
- `RUNPOD_MAX_READ_TEXT_BYTES` — defaults to 1 MiB.
- `RUNPOD_MAX_ARTIFACT_BYTES` — defaults to 512 MiB.
- `RUNPOD_STATUS_POLL_SECONDS` — defaults to 20 seconds where polling is needed.
- `RUNPOD_MAX_HOURLY_USD` — no global controller cap when unset.
- `RUNPOD_MAX_RUNTIME_MINUTES` — no global controller timeout when unset.
- `RUNPOD_TEMPLATE_ID`
- `RUNPOD_CONTAINER_REGISTRY_AUTH_ID`

Detached compute is the default. A detached experiment is not terminated because a GitHub runner reaches the end of its lease. The supervisor hands monitoring to another GitHub runner while the RunPod Pod continues uninterrupted. The scheduled reconciler is a fallback.

## Typed compute runners

### `infra_smoke`

Checks the real RunPod environment without starting a scientific training stage:

- CUDA/PyTorch visibility
- GPU identity and VRAM report
- CPU and RAM
- mounted `/workspace` capacity
- read/write access to control metadata
- Ardor model/config imports from image code
- presence of common persistent Ardor paths

Example: `runpod/examples/infra-smoke.json`.

### `ardor_promptgen`

Invokes the existing `Cerebrum/Cortex/neural_plasticity_training.py` through the Hephaestus adapter. Enabled stages are exactly the stages currently exposed by that trainer:

- `lm_base`
- `stabilize`
- `sft`

A real training job must explicitly provide `task.tokenizer`. The control plane will not guess a tokenizer version.

Example shape:

```json
{
  "version": 1,
  "id": "ardor-stabilize-001",
  "kind": "compute",
  "task": {
    "runner": "ardor_promptgen",
    "stage": "stabilize",
    "tokenizer": "/workspace/Ardor/tokenizer_v9.json",
    "resume": "/workspace/Ardor/runs/ardor_promptgen/checkpoints/ckpt_full_latest.pt",
    "train_tokens": "/workspace/Ardor/bin_dataset_20B/tokens.bin",
    "val_tokens": "/workspace/Ardor/bin_dataset_heldout_25M/tokens.bin",
    "seed": 1337
  },
  "gpu": {
    "type_ids": ["NVIDIA GeForce RTX 4090"],
    "count": 1,
    "cloud_type": "SECURE"
  },
  "lifecycle": {
    "mode": "detached"
  }
}
```

Do not copy that manifest into `runpod/jobs/` until the paths and scientific intent have been verified. A JSON file added under `runpod/jobs/` on `main` is an execution request.

The first version deliberately rejects remote overrides for `hidden_size`, `n_layers`, `n_heads`, `ff_mult`, and `ctx`. Those are architecture/context choices with checkpoint-compatibility consequences and should be changed only through reviewed Ardor code or an explicitly designed migration.

## S3 operations

User-facing storage jobs are read-only:

- `list`
- `head`
- `read_text`
- `download`

The controller itself can write/delete only its own metadata under:

- `ardor-control/active/`
- `ardor-control/history/`

Workers write status/logs under:

- `ardor-control/runs/<job-id>/<control-run-id>/`

Large datasets and checkpoints remain on the Network Volume; they do not need to pass through GitHub or chat.

## Lifecycle

1. A job manifest is validated by GitHub Actions.
2. The controller creates one allowed GPU Pod in the Network Volume's datacenter.
3. The Pod mounts the volume at `/workspace` and starts `scripts/runpod_worker.py` from `/opt/Ardor`.
4. The worker executes only a typed Ardor operation and streams stdout both to the Pod log and persistent `worker.log`.
5. The worker writes terminal `status.json` on completion or failure.
6. The GitHub supervisor/reconciler sees terminal status, terminates the Pod, archives the control record, and leaves scientific artifacts on the volume.

Manual termination is available through the `Ardor RunPod Reconciler` workflow by supplying a valid `control_run_id`.

## Image publishing

`.github/workflows/build-runpod-agent.yml` builds `docker/runpod-agent.Dockerfile` on pull requests and publishes on `main` to:

- `ghcr.io/<owner>/ardor-runpod:latest`
- `ghcr.io/<owner>/ardor-runpod:sha-<commit>`

The repository variable `RUNPOD_IMAGE_NAME` selects the image used by the controller.

## Operating rule

Keep code changes and execution requests separate. Merge/build the code image first; then create or dispatch the scientific `runpod/jobs/*.json` manifest. That prevents a new scientific job from racing an image build for the same commit.
