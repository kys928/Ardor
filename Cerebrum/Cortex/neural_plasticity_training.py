from __future__ import annotations
import argparse, os, sys, math, random, glob, time, re, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable, Callable
from torch.amp import GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer


# --------------------------- IO utils ---------------------------

def read_texts_from_glob(pattern: str, limit: int | None = None) -> List[str]:
    paths = sorted(glob.glob(pattern, recursive=False))
    out = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
                if txt:
                    out.append(txt)
        except Exception:
            pass
        if limit and len(out) >= limit:
            break
    random.shuffle(out)
    return out


# --------------------------- Token datasets ---------------------------

class TokenChunkDataset(Dataset):
    """Chunk raw texts to contiguous token blocks; returns (x,y) LongTensors."""

    def __init__(self, texts: List[str], token_json: str, ctx_len: int = 1024):
        self.tok = Tokenizer.from_file(token_json)
        ids: List[int] = []
        for t in texts:
            seq = self.tok.encode(t).ids
            if not seq:
                continue
            ids.extend(seq)
        self.ctx_len = int(ctx_len)
        self.blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(0, max(0, len(ids) - 1), self.ctx_len):
            chunk = ids[i:i + self.ctx_len + 1]
            if len(chunk) >= 2:
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.blocks.append((x, y))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


class TokenShardDataset(Dataset):
    """Replay directly from .pt token shards that contain 1D/2D token tensors or common dict/list formats."""

    def __init__(self, shard_paths: Iterable[Path], ctx_len: int = 1024):
        self.blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ctx_len = int(ctx_len)
        for p in shard_paths:
            try:
                obj = torch.load(p, map_location="cpu")
            except Exception:
                continue
            seqs: List[torch.Tensor] = []
            if isinstance(obj, torch.Tensor):
                if obj.ndim == 1:
                    seqs = [obj.long()]
                elif obj.ndim == 2:
                    seqs = [row.long() for row in obj]
            elif isinstance(obj, (list, tuple)):
                for it in obj:
                    if isinstance(it, torch.Tensor) and it.ndim == 1:
                        seqs.append(it.long())
                    elif isinstance(it, (list, tuple)) and it and isinstance(it[0], int):
                        seqs.append(torch.tensor(it, dtype=torch.long))
            elif isinstance(obj, dict):
                for k in ("rows", "seqs", "sequences", "input_ids_list", "dataset", "input_ids"):
                    if k in obj and isinstance(obj[k], (list, tuple)):
                        for it in obj[k]:
                            if isinstance(it, torch.Tensor) and it.ndim == 1:
                                seqs.append(it.long())
                            elif isinstance(it, (list, tuple)) and it and isinstance(it[0], int):
                                seqs.append(torch.tensor(it, dtype=torch.long))
            for ids in seqs:
                ids = ids.view(-1)
                if ids.numel() < 2:
                    continue
                for i in range(0, max(0, ids.numel() - 1), ctx_len):
                    chunk = ids[i:i + ctx_len + 1]
                    if chunk.numel() >= 2:
                        self.blocks.append((chunk[:-1], chunk[1:]))
        if not self.blocks:
            raise RuntimeError("No valid token blocks found in .pt shards.")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        x, y = self.blocks[i]
        return x.clone(), y.clone()


# --------------------------- Collate ---------------------------

def make_pad_collate(pad_id: int | None, ignore_idx: int) -> Callable:
    pad_token = 0 if pad_id is None else int(pad_id)

    def collate(batch):
        max_len = max(x.size(0) for x, _ in batch)
        B = len(batch)
        X = torch.full((B, max_len), pad_token, dtype=torch.long)
        Y = torch.full((B, max_len), ignore_idx, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            n = x.size(0)
            X[i, :n] = x
            Y[i, :n] = y
        return X, Y

    return collate


# --------------------------- Regularizers & UL ---------------------------

def l2sp_loss(model: nn.Module, ref_state: Dict[str, torch.Tensor], strength: float = 1e-3,
              exclude_norm_bias: bool = True) -> torch.Tensor:
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    for n, p in model.named_parameters():
        if (not p.requires_grad) or (n not in ref_state):
            continue
        r = ref_state[n].to(device)
        if p.shape != r.shape:
            continue
        if exclude_norm_bias and p.dim() == 1 and ("norm" in n.lower() or n.endswith("bias")):
            continue
        loss = loss + strength * torch.mean((p - r) ** 2)
    return loss


def build_ban_ids(tok: Tokenizer, ban_strings: str) -> Optional[torch.Tensor]:
    ban_set = set()
    if not ban_strings:
        return None
    parts = [s.strip() for s in ban_strings.split(';') if s.strip()]
    for s in parts:
        ids = tok.encode(s).ids
        for tid in ids:
            ban_set.add(int(tid))
    if not ban_set:
        return None
    return torch.tensor(sorted(ban_set), dtype=torch.long)


def unlikelihood_loss_masked(
        logits: torch.Tensor, targets: torch.Tensor, ban_ids: Optional[torch.Tensor], ignore_idx: int
) -> torch.Tensor:
    """UL with masking: ignore `ignore_idx`; and never punish if the gold token itself is banned."""
    if ban_ids is None or ban_ids.numel() == 0:
        return logits.new_tensor(0.0)

    device = logits.device
    B, T, V = logits.shape

    ban_ids = ban_ids.to(device)
    ban_ids = ban_ids[(ban_ids >= 0) & (ban_ids < V)]
    if ban_ids.numel() == 0:
        return logits.new_tensor(0.0)

    non_ignored = (targets != ignore_idx)
    gold_is_banned = torch.isin(targets.clamp(min=0), ban_ids)
    mask = (non_ignored & (~gold_is_banned)).float()  # [B,T]
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    K = ban_ids.numel()
    gather_idx = ban_ids.view(1, 1, K).expand(B, T, K)  # [B,T,K]
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p_bad = torch.exp(torch.gather(logp, dim=-1, index=gather_idx))  # [B,T,K]

    ul_tok = -torch.log(torch.clamp(1.0 - p_bad, min=1e-6))
    ul_tok = ul_tok * mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * K
    return ul_tok.sum() / denom


# --------------------------- Eval helpers ---------------------------

@torch.no_grad()
def student_token_nll(student: nn.Module, tok_json: str, texts: List[str], device: str = 'cuda',
                      cap: Optional[int] = None, max_len: int = 1024) -> List[Tuple[float, str]]:
    T = Tokenizer.from_file(tok_json)
    student.eval()
    out = []
    it = texts if cap is None else texts[:cap]
    for t in it:
        ids = T.encode(t).ids
        if not ids or len(ids) < 4:
            continue
        ids = ids[:max_len]
        inp = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
        tgt = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(device)
        logits = student(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='mean')
        out.append((loss.item(), t))
    return sorted(out, key=lambda x: x[0], reverse=True)


# --------------------------- Dream generator ---------------------------

@torch.no_grad()
def generate_dreams(teacher, teacher_tok, seeds: List[str], max_new=240, temp=0.7, top_p=0.9, rep_pen=1.1,
                    device='cuda') -> List[str]:
    dreams = []
    for s in seeds:
        prompt = f"Paraphrase and extend with rigorous reasoning, preserving the philosophical stance.\n\n{s}\n\nAnswer:"
        ids = teacher_tok(prompt, return_tensors='pt').to(device)
        out = teacher.generate(
            **ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p,
            repetition_penalty=rep_pen, pad_token_id=teacher_tok.eos_token_id
        )
        gen = teacher_tok.decode(out[0], skip_special_tokens=True)
        dreams.append(gen)
    return dreams


# --------------------------- Contrastive ranking ---------------------------

def continuation_score(logits: torch.Tensor, targets: torch.Tensor, ignore_idx: int) -> torch.Tensor:
    """Average log-prob per token for each sample in batch; safe for ignored positions."""
    logp = logits.log_softmax(dim=-1)  # [B,T,V]
    V = logits.size(-1)
    mask = (targets != ignore_idx)  # [B,T]
    safe_idx = torch.where(mask, targets, torch.zeros_like(targets)).clamp_(0, V - 1)
    tok_lp = torch.gather(logp, dim=-1, index=safe_idx.unsqueeze(-1)).squeeze(-1)  # [B,T]
    denom = mask.sum(dim=1).clamp_min(1)
    return (tok_lp * mask).sum(dim=1) / denom


def inbatch_ranking_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float, ignore_idx: int) -> torch.Tensor:
    if logits.size(0) < 2:
        return logits.new_tensor(0.0)
    pos = continuation_score(logits, targets, ignore_idx)
    neg_t = torch.roll(targets, shifts=1, dims=0)
    neg = continuation_score(logits, neg_t, ignore_idx)
    return (margin - pos + neg).clamp_min(0.0).mean()


# --------------------------- Resume helpers ---------------------------

def find_latest_epoch_ckpt(out_dir: Path) -> tuple[Optional[Path], int]:
    latest_path, latest_num = None, 0
    for p in out_dir.glob("epoch_*.pt"):
        m = re.search(r"epoch_(\d+)", p.stem)
        if m:
            n = int(m.group(1))
            if n > latest_num:
                latest_num = n
                latest_path = p
    return latest_path, latest_num  # num corresponds to finished epoch count


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # Hard-coded defaults (overrideable)
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--base', default=str(repo_root))
    ap.add_argument('--student_ckpt', default='artifacts/models/Ardor_Orion.pt')
    ap.add_argument('--student_tokenizer', default='Cerebrum/ProjectTokenizer/ardor_tokenizer/tokenizer_v9.json')
    ap.add_argument('--philo_glob', default='data/Philosophy/*.txt')
    ap.add_argument('--dialog_glob', default='data/Conversations/*.txt')
    ap.add_argument('--heldout', default='data/Philosophy_Heldout/*.txt')
    ap.add_argument('--teacher_id', default='mistralai/Mistral-7B-v0.1')
    ap.add_argument('--teacher_adapter_dir', default='runs/mistral_pipeline/20250831_210439/phase1_finetune')
    ap.add_argument('--out_dir', default='artifacts/models/rem_omega')
    ap.add_argument('--out_name', default='Ardor_Orion_REM.pt')

    # REM params
    ap.add_argument('--rem_epochs', type=int, default=2)
    ap.add_argument('--rem_lr', type=float, default=1.2e-4)
    ap.add_argument('--rem_batch', type=int, default=8)
    ap.add_argument('--grad_acc', type=int, default=2)
    ap.add_argument('--rem_dropout_max', type=float, default=0.2)
    ap.add_argument('--rem_dropout_min', type=float, default=0.05)
    ap.add_argument('--max_steps_per_epoch', type=int, default=0, help='Optional cap on steps per epoch (0 = no cap).')

    # Loss knobs
    ap.add_argument('--l2sp_lambda', type=float, default=5e-4)
    ap.add_argument('--l2sp_exclude_norm_bias', action='store_true')
    ap.add_argument('--ul_weight', type=float, default=0.2)
    ap.add_argument('--ban_strings', type=str,
                    default='PROJECT GUTENBERG;EBOOK;PUBLIC DOMAIN;***START OF;User:;Assistant:')
    ap.add_argument('--rank_lambda', type=float, default=0.05)
    ap.add_argument('--rank_margin', type=float, default=0.1)

    # Dreams
    ap.add_argument('--dream_hard_cap', type=int, default=64)
    ap.add_argument('--dream_rand_cap', type=int, default=64)
    ap.add_argument('--dream_max_new', type=int, default=240)
    ap.add_argument('--dream_temp', type=float, default=0.7)
    ap.add_argument('--dream_top_p', type=float, default=0.9)
    ap.add_argument('--dream_rep_pen', type=float, default=1.1)
    ap.add_argument('--dream_every', type=int, default=2)
    ap.add_argument('--dream_ce_lambda', type=float, default=0.4)
    ap.add_argument('--dialog_ratio', type=float, default=0.5, help='Fraction of random seeds drawn from dialogues.')

    # Averaging
    ap.add_argument('--swa', action='store_true', default=True)
    ap.add_argument('--swa_start_frac', type=float, default=0.5)

    # Resume controls
    ap.add_argument('--resume_from', type=str, default='', help='Path to a student epoch_XX.pt to resume from.')
    ap.add_argument('--resume_swa_from', type=str, default='', help='Path to an SWA snapshot to seed SWA from.')
    ap.add_argument('--no_auto_resume', action='store_true', help='Disable auto-resume discovery in out_dir.')

    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--cuda_device', type=int, default=0)
    args = ap.parse_args()

    # ---- Repro + CUDA-friendly defaults ----
    random.seed(args.seed);
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    base = Path(args.base).resolve()
    out_dir = (base / args.out_dir).resolve();
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    # tokenizer
    tok_path = (base / args.student_tokenizer).resolve()
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))

    # robust PAD id lookup (avoid treating 0 as falsy)
    def _first_id(names: List[str]) -> Optional[int]:
        for n in names:
            i = tok.token_to_id(n)
            if i is not None:
                return int(i)
        return None

    PAD_ID = _first_id(['<pad>', '<PAD>', '[PAD]'])
    IGNORE_IDX = PAD_ID if PAD_ID is not None else -100

    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler('cuda', enabled=use_scaler)

    # Load student (base)
    sys.path.append(str(base))
    sys.path.append(str((base / "Cerebrum" / "Cortex").resolve()))
    from broca_decoder import ArdorDecoder
    from ardor_config import ArdorConfig
    base_raw = torch.load((base / args.student_ckpt).resolve(), map_location='cpu')

    if isinstance(base_raw, dict):
        base_sd = base_raw.get('state_dict') or base_raw.get('model') or base_raw
        base_meta = dict(base_raw.get('config') or base_raw.get('meta') or {})
    else:
        base_sd = base_raw
        base_meta = {}

    if not isinstance(base_sd, dict):
        raise RuntimeError('Unsupported base checkpoint format for student initialization')

    if 'token_embed.weight' in base_sd:
        vocab = int(base_sd['token_embed.weight'].shape[0])
        hidden = int(base_sd['token_embed.weight'].shape[1])
    else:
        vocab = int(tok.get_vocab_size())
        hidden = int(base_meta.get('hidden_size') or base_meta.get('hidden') or 768)
    try:
        layers = int(base_meta.get('n_layers') or base_meta.get('layers') or max(int(k.split('.')[1]) for k in base_sd.keys() if k.startswith('blocks.') and '.attn.q.' in k) + 1)
    except Exception:
        layers = int(base_meta.get('n_layers') or base_meta.get('layers') or 12)
    heads = int(base_meta.get('n_heads') or base_meta.get('heads') or (12 if hidden % 12 == 0 else 8))
    max_len = int(base_meta.get('max_len') or 1024)

    cfg = ArdorConfig(vocab_size=vocab, hidden_size=hidden, n_layers=layers, n_heads=heads, max_len=max_len, dropout=0.15, attn_dropout=0.15, resid_dropout=0.15)
    student = ArdorDecoder(cfg).to(device)
    student.load_state_dict(base_sd, strict=False)

    # Reference state for L2-SP (anchor = base checkpoint)
    ref_state = {k: v.clone().detach() for k, v in base_sd.items()}

    # ---- Data: philosophy + dialogues ----
    philo_texts = read_texts_from_glob(str(base / args.philo_glob))
    dialog_texts = read_texts_from_glob(str(base / args.dialog_glob)) if args.dialog_glob else []
    texts_all = (philo_texts or []) + (dialog_texts or [])

    collate_fn = make_pad_collate(PAD_ID, IGNORE_IDX)
    if texts_all:
        real_ds = TokenChunkDataset(texts_all, str(tok_path), ctx_len=1024)
        if len(real_ds) == 0:
            print('[warn] Texts found but produced 0 blocks after tokenization; will try .pt shards.')
            real_loader = None
        else:
            real_loader = DataLoader(
                real_ds, batch_size=args.rem_batch, shuffle=True, drop_last=True,
                collate_fn=collate_fn, pin_memory=True, persistent_workers=False
            )
    else:
        real_loader = None

    if real_loader is None:
        # fallback: try philosophy and conversation shard dirs
        shard_paths = []
        for d in [base / 'artifacts' / 'datasets' / 'Philosophy', base / 'artifacts' / 'datasets' / 'Conversations']:
            if d.is_dir():
                shard_paths.extend(sorted(d.rglob('*.pt')))
        if not shard_paths:
            raise SystemExit('❌ No data for REM: no .txt matched and no .pt shards found in artifacts/datasets.')
        print(f"[rem] Falling back to .pt shards: {len(shard_paths)} files")
        real_ds = TokenShardDataset(shard_paths, ctx_len=1024)
        real_loader = DataLoader(
            real_ds, batch_size=args.rem_batch, shuffle=True, drop_last=True,
            collate_fn=collate_fn, pin_memory=True, persistent_workers=False
        )

    held_texts = read_texts_from_glob(str(base / args.heldout))

    # ---- Teacher (dreams) ----
    teacher = None;
    teacher_tok = None
    if args.teacher_id:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            teacher_tok = AutoTokenizer.from_pretrained(args.teacher_adapter_dir or args.teacher_id)
            base_model = AutoModelForCausalLM.from_pretrained(args.teacher_id, quantization_config=bnb_cfg,
                                                              device_map='auto')
            if args.teacher_adapter_dir and Path(base / args.teacher_adapter_dir).exists():
                teacher = PeftModel.from_pretrained(base_model, str(base / args.teacher_adapter_dir)).to(device).eval()
            else:
                teacher = base_model.to(device).eval()
        except Exception as e:
            print('[warn] teacher load failed:', e)
            teacher = None

    # ---- UL ban ids (filtered to vocab range) ----
    ban_ids = build_ban_ids(tok, args.ban_strings)
    vocab_size_for_filter = student.token_embed.weight.size(0) if hasattr(student,
                                                                          "token_embed") else tok.get_vocab_size()
    if ban_ids is not None:
        ban_ids = ban_ids[(ban_ids >= 0) & (ban_ids < vocab_size_for_filter)]
        if ban_ids.numel() == 0:
            ban_ids = None
    print('[info] ban_ids count:', 0 if ban_ids is None else ban_ids.numel())

    # ---- Optimizer & scheduler (scheduler counts optimizer updates, not micro-steps) ----
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.rem_lr, weight_decay=0.01)

    steps_per_epoch = len(real_loader)
    eff_steps_per_epoch = min(steps_per_epoch,
                              args.max_steps_per_epoch) if args.max_steps_per_epoch > 0 else steps_per_epoch
    updates_per_epoch = math.ceil(eff_steps_per_epoch / max(1, args.grad_acc))
    total_updates = max(1, args.rem_epochs * updates_per_epoch)

    def cosine_with_warmup(steps, warm):
        def fn(s):
            if s < warm: return float(s) / float(max(1, warm))
            prog = (s - warm) / max(1, (steps - warm))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return fn

    warmup = max(1, int(0.05 * total_updates))

    # ---- SWA ----
    use_swa = bool(args.swa)
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, update_bn
        swa_model = AveragedModel(student)
        swa_start = max(0, int(args.rem_epochs * args.swa_start_frac))
    else:
        swa_model = None;
        swa_start = int(1e9)

    # ---- Resume (model + SWA + LR schedule fast-forward) ----
    start_epoch = 0
    resume_msg = ""
    explicit_resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    explicit_swa_path = Path(args.resume_swa_from).resolve() if args.resume_swa_from else None

    auto_resume_path, finished_epochs = (None, 0) if args.no_auto_resume else find_latest_epoch_ckpt(out_dir)
    if explicit_resume_path and explicit_resume_path.exists():
        ck_path = explicit_resume_path
        start_epoch = 0  # if user points to epoch_N.pt, we'll parse it below
    elif auto_resume_path:
        ck_path = auto_resume_path
        start_epoch = finished_epochs
    else:
        ck_path = None

    if ck_path is not None and ck_path.exists():
        resume_raw = torch.load(ck_path, map_location='cpu')
        resume_sd = resume_raw.get('state_dict') if isinstance(resume_raw, dict) else resume_raw
        if isinstance(resume_raw, dict) and isinstance(resume_raw.get('model'), dict) and not isinstance(resume_sd, dict):
            resume_sd = resume_raw.get('model')
        student.load_state_dict(resume_sd, strict=False)
        # parse epoch number from filename
        m = re.search(r"epoch_(\d+)", ck_path.stem)
        if m:
            start_epoch = int(m.group(1))
        resume_msg += f"[resume] Loaded student from {ck_path.name}; starting at epoch {start_epoch + 1}/{args.rem_epochs}\n"

    # SWA resume (seed averaged weights)
    if use_swa:
        if explicit_swa_path and explicit_swa_path.exists():
            try:
                swa_raw = torch.load(explicit_swa_path, map_location='cpu')
                swa_sd = swa_raw.get('state_dict') if isinstance(swa_raw, dict) else swa_raw
                if isinstance(swa_raw, dict) and isinstance(swa_raw.get('model'), dict) and not isinstance(swa_sd, dict):
                    swa_sd = swa_raw.get('model')
                swa_model.load_state_dict(swa_sd, strict=False)
                resume_msg += f"[resume] Loaded SWA snapshot from {explicit_swa_path.name}\n"
            except Exception as e:
                print(f"[warn] failed to load SWA from {explicit_swa_path}: {e}")
        else:
            swa_default = out_dir / "swa_snapshot.pt"
            if swa_default.exists():
                try:
                    swa_raw = torch.load(swa_default, map_location='cpu')
                    swa_sd = swa_raw.get('state_dict') if isinstance(swa_raw, dict) else swa_raw
                    if isinstance(swa_raw, dict) and isinstance(swa_raw.get('model'), dict) and not isinstance(swa_sd, dict):
                        swa_sd = swa_raw.get('model')
                    swa_model.load_state_dict(swa_sd, strict=False)
                    resume_msg += f"[resume] Loaded SWA snapshot from swa_snapshot.pt\n"
                except Exception as e:
                    print(f"[warn] failed to load SWA from swa_snapshot.pt: {e}")

    # Now that we know how many epochs were finished, fast-forward LR schedule
    prev_updates = start_epoch * updates_per_epoch

    # Ensure param_groups have initial_lr when resuming without optimizer state
    for pg in opt.param_groups:
        pg.setdefault('initial_lr', pg['lr'])

    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(total_updates, warmup),
                                              last_epoch=prev_updates - 1)

    if resume_msg:
        print(resume_msg.strip())

    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)

    # banner
    print(f"[cfg] epochs={args.rem_epochs} batch={args.rem_batch} grad_acc={args.grad_acc} "
          f"lr={args.rem_lr:.2e} swa={args.swa} l2sp={args.l2sp_lambda} ul={args.ul_weight} rank={args.rank_lambda} "
          f"steps/epoch={steps_per_epoch} (effective {eff_steps_per_epoch}) updates/epoch={updates_per_epoch}")



    def _checkpoint_payload(model_obj):
        mm = model_obj.module if hasattr(model_obj, 'module') else model_obj
        sd = mm.state_dict()
        if hasattr(mm, 'model_config'):
            cfg = mm.model_config()
        elif hasattr(mm, 'cfg') and hasattr(mm.cfg, 'to_dict'):
            cfg = mm.cfg.to_dict()
        else:
            cfg = {}
        arch = cfg.get('arch') if isinstance(cfg, dict) else None
        if not arch:
            arch = type(mm).__name__
        tok_sha = ''
        try:
            tok_sha = hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
        except Exception:
            tok_sha = ''
        payload = {
            'state_dict': sd,
            'model': sd,
            'arch': arch,
            'config': cfg,
            'tokenizer_path': str(tok_path),
            'tokenizer_vocab_size': int(tok.get_vocab_size()),
            'tokenizer_sha256': tok_sha,
            'positional_encoding': cfg.get('positional_encoding') if isinstance(cfg, dict) else None,
            'meta': cfg if isinstance(cfg, dict) else {},
        }
        return payload

    # --------------------------- TRAIN ---------------------------
    global_step = 0
    for ep in range(start_epoch, args.rem_epochs):
        # dropout schedule
        drop_p = max(
            0.0,
            args.rem_dropout_max - (float(ep) / max(1, args.rem_epochs - 1)) * (
                        args.rem_dropout_max - args.rem_dropout_min)
        )
        for m in student.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop_p
        print(f"[epoch {ep + 1}] dropout={drop_p:.3f}")

        student.train()
        total_loss = 0.0
        loss_ema = None
        t0 = time.time()
        tok_count = 0
        steps_in_epoch = len(real_loader)
        steps_cap = eff_steps_per_epoch

        # Seed pool for dreams
        seed_pool = texts_all
        hard_seeds = []
        if teacher is not None and args.dream_hard_cap > 0 and seed_pool:
            ranked = student_token_nll(student, str(tok_path), seed_pool, device=device.type,
                                       cap=args.dream_hard_cap * 4, max_len=1024)
            hard_seeds = [t for _, t in ranked[:args.dream_hard_cap]]

        rand_dialog_n = int(args.dream_rand_cap * args.dialog_ratio)
        rand_philo_n = max(0, args.dream_rand_cap - rand_dialog_n)
        rand_seeds = []
        if dialog_texts:
            rand_seeds.extend(random.sample(dialog_texts, k=min(rand_dialog_n, len(dialog_texts))))
        if philo_texts:
            rand_seeds.extend(random.sample(philo_texts, k=min(rand_philo_n, len(philo_texts))))
        seed_texts = hard_seeds + rand_seeds

        dreams = []
        if teacher is not None and seed_texts:
            print(f"[dreams] generating {len(seed_texts)} dreams…")
            try:
                dreams = generate_dreams(teacher, teacher_tok, seed_texts, max_new=args.dream_max_new,
                                         temp=args.dream_temp, top_p=args.dream_top_p, rep_pen=args.dream_rep_pen,
                                         device=device.type)
            except Exception as e:
                print('[warn] dream generation failed:', e);
                dreams = []
            torch.cuda.empty_cache()

        dream_loader = None
        if dreams:
            dream_ds = TokenChunkDataset(dreams, str(tok_path), ctx_len=1024)
            if len(dream_ds) > 0:
                dream_loader = DataLoader(
                    dream_ds, batch_size=args.rem_batch, shuffle=True, drop_last=True,
                    collate_fn=collate_fn, pin_memory=True, persistent_workers=False
                )
                dream_iter = iter(dream_loader)
            else:
                dream_iter = None
        else:
            dream_iter = None

        accum = 0
        for i, (xb, yb) in enumerate(real_loader, start=1):
            if steps_cap and i > steps_cap:
                break

            xb = xb.to(device, non_blocking=True);
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                logits = student(xb)
                loss = crit(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

                # Ranking (light) on real batch
                if args.rank_lambda > 0:
                    rloss = inbatch_ranking_loss(logits, yb, margin=float(args.rank_margin), ignore_idx=IGNORE_IDX)
                    loss = loss + float(args.rank_lambda) * rloss

                # UL on real (masked)
                if args.ul_weight > 0 and ban_ids is not None:
                    ul_real = unlikelihood_loss_masked(logits, yb, ban_ids, ignore_idx=IGNORE_IDX)
                    loss = loss + float(args.ul_weight) * ul_real

                # L2-SP regularizer
                if args.l2sp_lambda > 0:
                    loss = loss + l2sp_loss(student, ref_state, strength=args.l2sp_lambda,
                                            exclude_norm_bias=args.l2sp_exclude_norm_bias)

                # Mix dream batch periodically
                dream_tok = 0
                if dream_iter is not None and (global_step % max(1, args.dream_every) == 0):
                    try:
                        dx, dy = next(dream_iter)
                    except StopIteration:
                        dream_iter = iter(dream_loader);
                        dx, dy = next(dream_iter)
                    dx = dx.to(device, non_blocking=True);
                    dy = dy.to(device, non_blocking=True)
                    d_logits = student(dx)
                    d_loss = crit(d_logits.reshape(-1, d_logits.size(-1)), dy.reshape(-1))
                    if args.rank_lambda > 0:
                        d_rloss = inbatch_ranking_loss(d_logits, dy, margin=float(args.rank_margin),
                                                       ignore_idx=IGNORE_IDX)
                        d_loss = d_loss + float(args.rank_lambda) * d_rloss
                    if args.ul_weight > 0 and ban_ids is not None:
                        ul_d = unlikelihood_loss_masked(d_logits, dy, ban_ids, ignore_idx=IGNORE_IDX)
                        d_loss = d_loss + float(args.ul_weight) * ul_d
                    loss = loss + float(args.dream_ce_lambda) * d_loss
                    dream_tok = int((dy != IGNORE_IDX).sum().item())

            # grad accumulation
            if use_scaler:
                scaler.scale(loss / max(1, args.grad_acc)).backward()
            else:
                (loss / max(1, args.grad_acc)).backward()
            accum += 1

            # tokens processed this iteration
            tok_count += int((yb != IGNORE_IDX).sum().item()) + dream_tok

            if accum % max(1, args.grad_acc) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                if use_scaler:
                    scaler.step(opt);
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

                if use_swa and ep >= swa_start:
                    swa_model.update_parameters(student)

            # progress line
            loss_val = float(loss.item())
            loss_ema = loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * loss_val)
            elapsed = max(1e-3, time.time() - t0)
            tps = tok_count / elapsed
            lr = opt.param_groups[0]['lr']
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                mem_str = f"{alloc:.1f}/{total:.1f}G"
            else:
                mem_str = "cpu"
            print(f"\r[ep {ep + 1}/{args.rem_epochs}] step {i:4d}/{steps_cap or steps_in_epoch} | "
                  f"loss~{loss_ema:.4f} | lr {lr:.2e} | tok/s {tps:.0f} | mem {mem_str}",
                  end='', flush=True)

            total_loss += loss_val
            global_step += 1
        print()  # newline after epoch progress

        avg_loss = total_loss / max(1, min(steps_in_epoch, steps_cap) if steps_cap else steps_in_epoch)
        print(f"[epoch {ep + 1}] avg_loss={avg_loss:.4f}")

        # quick eval on heldout
        student.eval()
        ppl = None
        if held_texts:
            try:
                nlls = student_token_nll(student, str(tok_path), held_texts, device=device.type, cap=32, max_len=1024)
                if nlls:
                    ppl = math.exp(sum(x[0] for x in nlls) / max(1, len(nlls)))
            except Exception as e:
                print('[warn] quick eval failed:', e)
        print(f"[epoch {ep + 1}] quick heldout ppl={ppl}")

        # checkpoint
        ck = out_dir / f'epoch_{ep + 1:02d}.pt'
        torch.save(_checkpoint_payload(student), ck)
        print('[save] epoch ckpt ->', ck)

        if use_swa and swa_model is not None:
            torch.save(_checkpoint_payload(swa_model), out_dir / 'swa_snapshot.pt')

    # finalize SWA
    final_model = student
    if use_swa and swa_model is not None:
        try:
            from torch.optim.swa_utils import update_bn
            update_bn(real_loader, swa_model)
            final_model = swa_model
            print('[swa] finalized & BN updated')
        except Exception as e:
            print('[swa] BN update skipped:', e)
            final_model = student

    # save final
    final_path = out_path
    torch.save(_checkpoint_payload(final_model), final_path)
    print('[done] saved final REM model ->', final_path)


if __name__ == '__main__':
    main()
