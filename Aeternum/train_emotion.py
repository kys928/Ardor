#!/usr/bin/env python3
import os, argparse, time, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

# ----------------------------
# Datasets
# ----------------------------
class NpyMultiHot(Dataset):
    """For GoEmotions: X [N,H], Y [N,C] (multi-hot)"""
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.float32)
        assert self.X.shape[0] == self.Y.shape[0], "X/Y mismatch"
        assert self.Y.ndim == 2, "Y must be [N,C] multi-hot"
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

class NpyClass(Dataset):
    """For ED & MELD: X [N,H], y [N] int or [N,C] one-hot"""
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path).astype(np.float32)
        Y = np.load(y_path)
        assert self.X.shape[0] == Y.shape[0], "X/Y mismatch"
        if Y.ndim == 1:  # class indices
            self.y = Y.astype(np.int64)
            self.C = int(self.y.max()) + 1
        elif Y.ndim == 2:
            self.y = Y.astype(np.float32)
            self.C = Y.shape[1]
        else:
            raise ValueError("Unsupported Y shape")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def collate_mh(batch):
    X = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    Y = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    return X, Y

def collate_cls(batch):
    X = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    y = batch[0][1]
    if isinstance(y, np.ndarray) and y.ndim == 0:
        # indices
        Y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    else:
        # one-hot
        Y = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    return X, Y

# ----------------------------
# Model
# ----------------------------
class Trunk(nn.Module):
    def __init__(self, in_dim, hidden=512, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(drop),
        )
    def forward(self, x):  # x: [B,H]
        return self.net(x)

class EmotionHeadsMTL(nn.Module):
    def __init__(self, in_dim, c_goe, c_ed, c_meld, hidden=512, drop=0.1):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden=hidden, drop=drop)
        self.goe_head = nn.Linear(hidden, c_goe)   # logits (sigmoid later)
        self.ed_head  = nn.Linear(hidden, c_ed)    # logits for softmax
        self.meld_head= nn.Linear(hidden, c_meld)  # logits for softmax
        # Temperature parameters (calibration); initialize at 1.0
        self.register_buffer("T_goe", torch.tensor(1.0))
        self.register_buffer("T_ed", torch.tensor(1.0))
        self.register_buffer("T_meld", torch.tensor(1.0))

    def forward(self, x):
        h = self.trunk(x)
        lg_goe  = self.goe_head(h)   # [B,C_goe] (pre-sigmoid)
        lg_ed   = self.ed_head(h)    # [B,C_ed]  (pre-softmax)
        lg_meld = self.meld_head(h)  # [B,C_meld]
        return lg_goe, lg_ed, lg_meld

    # Calibrated inference helpers
    def probs_goe(self, logits):   # multi-label sigmoid with T
        return torch.sigmoid(logits / self.T_goe.clamp(min=1e-6))
    def probs_softmax(self, logits, head="ed"):
        T = self.T_ed if head == "ed" else self.T_meld
        return torch.softmax(logits / T.clamp(min=1e-6), dim=-1)

# ----------------------------
# Losses & utilities
# ----------------------------
def compute_pos_weight_goe(Y):  # Y [N,C] 0/1
    # pos_weight = (N - pos) / pos
    N, C = Y.shape
    pos = np.clip(Y.sum(axis=0), 1.0, None)
    neg = N - pos
    w = torch.tensor(neg / pos, dtype=torch.float32)
    return w

def compute_class_weight_indices(y_idx, C):
    counts = np.bincount(y_idx, minlength=C).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = (inv / inv.sum()) * C  # normalized but sums to C
    return torch.tensor(w, dtype=torch.float32)

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=0.0, reduction="mean", pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        if self.gamma <= 0:
            loss = bce
        else:
            p = torch.sigmoid(logits)
            pt = targets * p + (1 - targets) * (1 - p)
            loss = ((1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=0.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    def forward(self, logits, target):
        if target.ndim == 2:
            # one-hot -> indices
            target = target.argmax(dim=-1)
        if self.gamma <= 0:
            return self.ce(logits, target)
        # standard CE with focal modulator
        logp = torch.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        # gather logp for true class
        t = target.view(-1,1)
        logp_t = logp.gather(1, t).squeeze(1)
        p_t = p.gather(1, t).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            w_t = self.weight.gather(0, target)
            loss = loss * w_t
        if self.ls > 0:
            # add smoothed uniform term (approx)
            loss = (1 - self.ls) * loss + self.ls * (-logp.mean(dim=-1))
        return loss.mean()

# ----------------------------
# Calibration (temperature scaling)
# ----------------------------
def find_temperature_sigmoid(model, dl, device, max_iter=200, lr=0.05):
    # One global temperature for multi-label head.
    T = torch.tensor(float(model.T_goe.item()), requires_grad=True, device=device)
    opt = optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    def closure():
        opt.zero_grad()
        losses = []
        with torch.no_grad():
            pass
        for X,Y in dl:
            X = X.to(device); Y = Y.to(device)
            lg_goe, _, _ = model(X)
            loss = bce(lg_goe / T.clamp(min=1e-6), Y)
            losses.append(loss)
        total = torch.stack(losses).mean()
        total.backward()
        return total

    try:
        opt.step(closure)
        T_val = float(T.detach().clamp(min=1e-6).item())
    except Exception:
        T_val = 1.0
    model.T_goe = torch.tensor(T_val, device=device)

def find_temperature_softmax(model, dl, device, head="ed", max_iter=200, lr=0.05):
    Tbuf = model.T_ed if head == "ed" else model.T_meld
    T = torch.tensor(float(Tbuf.item()), requires_grad=True, device=device)
    opt = optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        losses = []
        for X,Y in dl:
            X = X.to(device)
            if Y.ndim == 2:
                Y = Y.argmax(dim=-1)
            Y = Y.to(device)
            _, lg_ed, lg_meld = model(X)
            logits = lg_ed if head == "ed" else lg_meld
            loss = ce(logits / T.clamp(min=1e-6), Y)
            losses.append(loss)
        total = torch.stack(losses).mean()
        total.backward()
        return total
    try:
        opt.step(closure)
        T_val = float(T.detach().clamp(min=1e-6).item())
    except Exception:
        T_val = 1.0
    if head == "ed":
        model.T_ed = torch.tensor(T_val, device=device)
    else:
        model.T_meld = torch.tensor(T_val, device=device)

# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def eval_metrics(model, dl_goe, dl_ed, dl_meld, device, thr_goe=0.5):
    model.eval()
    out = {}

    # GoEmotions macro-F1 (multi-label, fixed threshold)
    if dl_goe is not None:
        y_true = []; y_pred = []
        for X,Y in dl_goe:
            X = X.to(device); Y = Y.to(device)
            lg_goe, _, _ = model(X)
            P = (torch.sigmoid(lg_goe) >= thr_goe).float()
            y_true.append(Y.cpu()); y_pred.append(P.cpu())
        Yt = torch.cat(y_true, 0).numpy(); Yp = torch.cat(y_pred, 0).numpy()
        eps = 1e-8
        f1_per_class = []
        for c in range(Yt.shape[1]):
            yt = Yt[:,c]; yp = Yp[:,c]
            tp = (yt*yp).sum()
            fp = ((1-yt)*yp).sum()
            fn = (yt*(1-yp)).sum()
            f1 = (2*tp)/(2*tp+fp+fn+eps)
            f1_per_class.append(float(f1))
        out["goe_macro_f1"] = float(np.mean(f1_per_class))

    # ED accuracy / macro-F1
    if dl_ed is not None:
        y_true = []; y_pred = []
        for X,Y in dl_ed:
            X = X.to(device)
            if Y.ndim == 2:
                Y = Y.argmax(dim=-1)
            Y = Y.to(device)
            _, lg_ed, _ = model(X)
            P = lg_ed.argmax(dim=-1)
            y_true.append(Y.cpu()); y_pred.append(P.cpu())
        yt = torch.cat(y_true, 0).numpy(); yp = torch.cat(y_pred, 0).numpy()
        acc = (yt == yp).mean()
        # macro-F1
        C = int(max(yt.max(), yp.max())) + 1
        f1s = []
        for c in range(C):
            tp = int(((yp==c) & (yt==c)).sum())
            fp = int(((yp==c) & (yt!=c)).sum())
            fn = int(((yp!=c) & (yt==c)).sum())
            f1 = 0.0 if (2*tp+fp+fn)==0 else (2*tp)/(2*tp+fp+fn)
            f1s.append(f1)
        out["ed_acc"] = float(acc)
        out["ed_macro_f1"] = float(np.mean(f1s))

    # MELD accuracy / macro-F1
    if dl_meld is not None:
        y_true = []; y_pred = []
        for X,Y in dl_meld:
            X = X.to(device)
            if Y.ndim == 2:
                Y = Y.argmax(dim=-1)
            Y = Y.to(device)
            _, _, lg_meld = model(X)
            P = lg_meld.argmax(dim=-1)
            y_true.append(Y.cpu()); y_pred.append(P.cpu())
        yt = torch.cat(y_true, 0).numpy(); yp = torch.cat(y_pred, 0).numpy()
        acc = (yt == yp).mean()
        C = int(max(yt.max(), yp.max())) + 1
        f1s = []
        for c in range(C):
            tp = int(((yp==c) & (yt==c)).sum())
            fp = int(((yp==c) & (yt!=c)).sum())
            fn = int(((yp!=c) & (yt==c)).sum())
            f1 = 0.0 if (2*tp+fp+fn)==0 else (2*tp)/(2*tp+fp+fn)
            f1s.append(f1)
        out["meld_acc"] = float(acc)
        out["meld_macro_f1"] = float(np.mean(f1s))

    return out

# ----------------------------
# Training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--save", default="emotion_heads_mtl.pt")

    # GoEmotions (multi-label)
    ap.add_argument("--goe-train-X", required=True)
    ap.add_argument("--goe-train-Y", required=True)
    ap.add_argument("--goe-val-X",   required=True)
    ap.add_argument("--goe-val-Y",   required=True)

    # EmpatheticDialogues (single-label)
    ap.add_argument("--ed-train-X", required=True)
    ap.add_argument("--ed-train-Y", required=True)
    ap.add_argument("--ed-val-X",   required=True)
    ap.add_argument("--ed-val-Y",   required=True)

    # MELD (single-label)
    ap.add_argument("--meld-train-X", required=True)
    ap.add_argument("--meld-train-Y", required=True)
    ap.add_argument("--meld-val-X",   required=True)
    ap.add_argument("--meld-val-Y",   required=True)

    # Loss schedule
    ap.add_argument("--stage-ratio", type=str, default="0.4,0.4,0.2", help="fractions for stages 1/2/3")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--focal-gamma-max", type=float, default=2.0)
    ap.add_argument("--focal-lambda-max", type=float, default=0.5)

    args = ap.parse_args()
    device = args.device

    # Datasets & loaders
    goe_tr = NpyMultiHot(args.goe_train_X, args.goe_train_Y)
    goe_va = NpyMultiHot(args.goe_val_X, args.goe_val_Y)
    ed_tr  = NpyClass(args.ed_train_X, args.ed_train_Y)
    ed_va  = NpyClass(args.ed_val_X, args.ed_val_Y)
    meld_tr= NpyClass(args.meld_train_X, args.meld_train_Y)
    meld_va= NpyClass(args.meld_val_X, args.meld_val_Y)

    H = goe_tr.X.shape[1]
    C_goe = goe_tr.Y.shape[1]
    C_ed  = ed_tr.C
    C_meld= meld_tr.C

    # Class weights / pos weights
    pos_w_goe = compute_pos_weight_goe(goe_tr.Y)        # [C_goe]
    if ed_tr.y.ndim == 1:
        w_ed = compute_class_weight_indices(ed_tr.y, C_ed)
    else:
        w_ed = compute_class_weight_indices(ed_tr.y.argmax(axis=1), C_ed)
    if meld_tr.y.ndim == 1:
        w_meld = compute_class_weight_indices(meld_tr.y, C_meld)
    else:
        w_meld = compute_class_weight_indices(meld_tr.y.argmax(axis=1), C_meld)

    dl_goe_tr = DataLoader(goe_tr, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_mh)
    dl_ed_tr  = DataLoader(ed_tr,  batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_cls)
    dl_meld_tr= DataLoader(meld_tr,batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_cls)

    dl_goe_va = DataLoader(goe_va, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_mh)
    dl_ed_va  = DataLoader(ed_va,  batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_cls)
    dl_meld_va= DataLoader(meld_va,batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_cls)

    # Model & opt
    model = EmotionHeadsMTL(in_dim=H, c_goe=C_goe, c_ed=C_ed, c_meld=C_meld,
                            hidden=args.hidden, drop=args.dropout).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss objects (configured per stage)
    def make_losses(gamma, ls):
        loss_goe = FocalBCEWithLogitsLoss(gamma=gamma, pos_weight=pos_w_goe.to(device))
        loss_ed  = FocalCrossEntropy(gamma=gamma, weight=w_ed.to(device), label_smoothing=ls)
        loss_mld = FocalCrossEntropy(gamma=gamma, weight=w_meld.to(device), label_smoothing=ls)
        return loss_goe, loss_ed, loss_mld

    # Stage split
    r1, r2, r3 = [float(x) for x in args.stage_ratio.split(",")]
    rsum = max(1e-6, r1 + r2 + r3)
    e1 = int(round(args.epochs * (r1/rsum)))
    e2 = int(round(args.epochs * (r2/rsum)))
    e3 = args.epochs - e1 - e2
    stages = [(e1, "warmup"), (e2, "focal"), (e3, "calibration")]

    best_score = -1.0
    best_state = None

    for si, (num_ep, name) in enumerate(stages, start=1):
        if num_ep <= 0:
            continue
        print(f"\n== Stage {si}/{len(stages)}: {name} for {num_ep} epochs ==")

        for ep in range(num_ep):
            model.train()
            t0 = time.time()

            # Ramp focal gamma & lambda across stages 2..3
            if name == "warmup":
                gamma = 0.0
                ls = args.label_smoothing
                lam = 0.0
            else:
                # progress 0..1 within (stages 2..3)
                total_remaining = sum(n for j,(n,_) in enumerate(stages) if j >= si-1)
                done_before = sum(n for j,(n,_) in enumerate(stages) if j < si-1)
                done_now = done_before + ep + 1
                prog = min(1.0, max(0.0, done_now / max(1, args.epochs)))
                gamma = args.focal_gamma_max * prog
                ls = max(0.0, args.label_smoothing * (1.0 - 0.5*prog))
                lam = args.focal_lambda_max * prog

            loss_goe_fn, loss_ed_fn, loss_mld_fn = make_losses(gamma, ls)

            it_goe = cycle(dl_goe_tr)
            it_ed  = cycle(dl_ed_tr)
            it_mld = cycle(dl_meld_tr)
            # Define steps per epoch roughly by the largest loader
            steps = max(len(dl_goe_tr), len(dl_ed_tr), len(dl_meld_tr))

            run_loss = 0.0
            for _ in range(steps):
                Xg,Yg = next(it_goe); Xe,Ye = next(it_ed); Xm,Ym = next(it_mld)
                Xg,Yg = Xg.to(device), Yg.to(device)
                Xe,Ye = Xe.to(device), Ye.to(device)
                Xm,Ym = Xm.to(device), Ym.to(device)

                lg_goe, lg_ed, lg_meld = model(Xg)  # trunk re-used; okay to compute once per batch, but shapes differ
                # For ED/MELD we need their own forward; recompute with Xe/Xm:
                _, lg_ed, _   = model(Xe)
                _, _, lg_meld = model(Xm)

                base_goe = loss_goe_fn(lg_goe, Yg)
                base_ed  = loss_ed_fn(lg_ed, Ye)
                base_mld = loss_mld_fn(lg_meld, Ym)

                # Focal lambda blends are already inside the focal implementations; we optionally add a small extra penalty:
                loss = base_goe + base_ed + base_mld
                if lam > 0:
                    # small entropy regularization to improve calibration
                    with torch.no_grad():
                        pg = torch.sigmoid(lg_goe).clamp(1e-5, 1-1e-5)
                        pe = torch.softmax(lg_ed, dim=-1).clamp(1e-5, 1-1e-5)
                        pm = torch.softmax(lg_meld, dim=-1).clamp(1e-5, 1-1e-5)
                    H = -(pg*pg.log() + (1-pg)*(1-pg).log()).mean() \
                        -(pe*pe.log()).mean() -(pm*pm.log()).mean()
                    loss = loss + 0.01*lam*H

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += float(loss.item())

            sched.step()
            dt = time.time() - t0

            # ---- Validation metrics ----
            with torch.no_grad():
                metrics = eval_metrics(model, dl_goe_va, dl_ed_va, dl_meld_va, device)
            score = (
                0.34 * metrics.get("goe_macro_f1", 0.0) +
                0.33 * metrics.get("ed_macro_f1", 0.0) +
                0.33 * metrics.get("meld_macro_f1", 0.0)
            )

            print(f"ep {ep+1}/{num_ep} [{name}]  loss {run_loss/steps:.4f}  "
                  f"goe_f1 {metrics.get('goe_macro_f1',0):.3f}  "
                  f"ed_f1 {metrics.get('ed_macro_f1',0):.3f}  "
                  f"meld_f1 {metrics.get('meld_macro_f1',0):.3f}  "
                  f"score {score:.3f}  {dt:.1f}s")

            if score > best_score:
                best_score = score
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}

        # Stage 3: calibration on validation sets (optimize temperature only)
        if name == "calibration":
            print("Calibrating temperatures on validation sets...")
            model.eval()
            model.to(device)
            find_temperature_sigmoid(model, dl_goe_va, device)
            find_temperature_softmax(model, dl_ed_va, device, head="ed")
            find_temperature_softmax(model, dl_meld_va, device, head="meld")

            # Re-evaluate after calibration
            with torch.no_grad():
                metrics = eval_metrics(model, dl_goe_va, dl_ed_va, dl_meld_va, device)
            score = (
                0.34 * metrics.get("goe_macro_f1", 0.0) +
                0.33 * metrics.get("ed_macro_f1", 0.0) +
                0.33 * metrics.get("meld_macro_f1", 0.0)
            )
            print(f"post-calibration val: "
                  f"goe_f1 {metrics.get('goe_macro_f1',0):.3f}  "
                  f"ed_f1 {metrics.get('ed_macro_f1',0):.3f}  "
                  f"meld_f1 {metrics.get('meld_macro_f1',0):.3f}  "
                  f"score {score:.3f}")

            if score > best_score:
                best_score = score
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    # Save checkpoint
    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "dims": {"H": int(H), "C_goe": int(C_goe), "C_ed": int(C_ed), "C_meld": int(C_meld)},
        "temps": {
            "T_goe": float(model.T_goe.item()),
            "T_ed":  float(model.T_ed.item()),
            "T_meld":float(model.T_meld.item()),
        },
        "args": vars(args),
    }
    torch.save(ckpt, args.save)
    print(f"✅ Saved best checkpoint to {args.save}")

if __name__ == "__main__":
    main()
