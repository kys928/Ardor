import os, json, argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from Aeternum.Amygdala import Amygdala

# ---------------- Data ----------------
def hazard_features(text: str):
    t = (text or "").lower()
    return np.array([
        1.0 if any(w in t for w in ["suicide","kill myself","overdose"]) else 0.0,
        1.0 if any(w in t for w in ["bomb","explosive"]) else 0.0,
        1.0 if ("prescrib" in t or "diagnos" in t) else 0.0,
        1.0 if "!" in t else 0.0,
        1.0 if any(w in t for w in ["urgent","asap","now","panic","immediately"]) else 0.0,
        1.0 if t.isupper() and len(t) > 6 else 0.0,
        min(1.0, t.count("!")/3.0),
        min(1.0, t.count("?")/3.0),
    ], dtype=np.float32)

class AmyDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.items.append((
                    ex.get("text",""),
                    int(ex.get("label",0)),
                    int(ex.get("difficulty",3))
                ))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        txt, y, d = self.items[i]
        return hazard_features(txt), np.float32(y), np.int64(d)

def collate(batch):
    X = torch.tensor(np.stack([b[0] for b in batch], 0), dtype=torch.float32)  # [B,F]
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)               # [B]
    d = torch.tensor([b[2] for b in batch], dtype=torch.long)                  # [B]
    return X, y, d

# -------------- Metrics --------------
def bin_metrics(y_true, y_prob, thr=0.5):
    yp = (y_prob >= thr).astype(np.int32)
    tp = int(((yp==1) & (y_true==1)).sum()); tn = int(((yp==0) & (y_true==0)).sum())
    fp = int(((yp==1) & (y_true==0)).sum()); fn = int(((yp==0) & (y_true==1)).sum())
    prec = tp / max(1, (tp+fp)); rec = tp / max(1, (tp+fn))
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return {"precision":prec, "recall":rec, "f1":f1, "tp":tp, "tn":tn, "fp":fp, "fn":fn}

def roc_auc(y_true, y_prob):
    order = np.argsort(y_prob)
    y = y_true[order]
    P = y.sum(); N = len(y)-P
    if P==0 or N==0: return 0.5
    rank_sum = 0.0
    for i,yi in enumerate(y, start=1):
        if yi==1: rank_sum += i
    return float((rank_sum - (P*(P+1)/2)) / (P*N))

# -------------- Loss bits --------------
def fn_hinge(y_true, y_rate, margin):
    # penalize positives that don't reach margin
    mask = (y_true > 0.5).float()
    return ((margin - y_rate).clamp(min=0.0) * mask).mean()

def focal_binary(y_true, y_prob, gamma=2.0, eps=1e-6):
    p_t = y_true*y_prob + (1-y_true)*(1-y_prob)
    return ((1 - p_t + eps)**gamma * (-(y_true*torch.log(y_prob+eps) + (1-y_true)*torch.log(1-y_prob+eps)))).mean()

def sparsity(rate, lam=0.0):
    return lam * rate.mean()

def latency_loss(latency, y_true, lam_pos=0.2, lam_neg=0.1):
    """
    latency in [0,1] where 1=early spike, 0=no spike/very late.
    Encourage early spikes for positives, late/none for negatives.
    L_pos = lam_pos * (1 - latency) * y
    L_neg = lam_neg * latency * (1 - y)
    """
    return (lam_pos * (1.0 - latency) * y_true + lam_neg * latency * (1.0 - y_true)).mean()

# -------------- Helpers --------------
def make_sampler(dataset, pos_weight=0.6):
    labels = np.array([int(y) for _,y,_ in dataset.items])
    pos_idx = np.where(labels==1)[0]; neg_idx = np.where(labels==0)[0]
    if len(pos_idx)==0 or len(neg_idx)==0: return None
    w_pos = pos_weight / max(1,len(pos_idx))
    w_neg = (1-pos_weight) / max(1,len(neg_idx))
    w = np.zeros(len(labels), dtype=np.float32); w[pos_idx]=w_pos; w[neg_idx]=w_neg
    return WeightedRandomSampler(w, num_samples=len(labels), replacement=True)

def curriculum_subset(dataset, max_difficulty):
    idx = [i for i,(_,_,d) in enumerate(dataset.items) if d <= max_difficulty]
    class Subset(Dataset):
        def __init__(self, base, idx): self.base=base; self.idx=idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, j): return self.base[self.idx[j]]
        @property
        def items(self): return [self.base.items[i] for i in self.idx]
    return Subset(dataset, idx)

# -------------- Train/Eval --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="amygdala_snn_snn.pt")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--max-difficulty", type=int, default=5)
    ap.add_argument("--pos-sample-weight", type=float, default=0.6)

    # temporal / encoding
    ap.add_argument("--Ts", type=int, default=6)
    ap.add_argument("--rate-encode", action="store_true")
    ap.add_argument("--use-gru", action="store_true")
    ap.add_argument("--refrac", type=int, default=2)
    ap.add_argument("--tau-rise", type=float, default=2.0)
    ap.add_argument("--tau-decay", type=float, default=8.0)

    # aversive schedule
    ap.add_argument("--fn-beta-min", type=float, default=0.0)
    ap.add_argument("--fn-beta-max", type=float, default=2.5)
    ap.add_argument("--margin-min", type=float, default=0.5)
    ap.add_argument("--margin-max", type=float, default=0.8)
    ap.add_argument("--sparsity-lam", type=float, default=0.01)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--focal-lambda", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # latency loss
    ap.add_argument("--lat-pos", type=float, default=0.2, help="positive-class latency weight")
    ap.add_argument("--lat-neg", type=float, default=0.1, help="negative-class latency weight")

    # checkpoints
    ap.add_argument("--resume", help="resume from SNN checkpoint")
    ap.add_argument("--transfer-perc", help="transfer from old perceptron-only checkpoint")
    args = ap.parse_args()

    device = args.device
    tr = AmyDataset(args.train); va = AmyDataset(args.val)

    model = Amygdala(
        in_feats=8, h1=64, h2=32,
        Ts=args.Ts, rate_encode=args.rate_encode,
        use_gru=args.use_gru, gru_hidden=32,
        tau_rise=args.tau_rise, tau_decay=args.tau_decay,
        refrac_steps=args.refrac, device=device
    )

    # ---- Safe resume (only if shapes match) ----
    if args.resume and Path(args.resume).exists():
        try:
            sd = torch.load(args.resume, map_location=device)
            model.load_state_dict(sd, strict=True)
            print(f"Resumed from {args.resume}")
        except Exception as e:
            print(f"[warn] resume failed ({e}); starting fresh.")

    # ---- Safe perceptron transfer (skip if incompatible) ----
    if args.transfer_perc and Path(args.transfer_perc).exists():
        try:
            old = torch.load(args.transfer_perc, map_location=device)
            mapped = {}
            for k,v in old.items():
                if "0.weight" in k: mapped["fc1.weight"] = v
                elif "0.bias" in k: mapped["fc1.bias"] = v
                elif "2.weight" in k: mapped["fc2.weight"] = v
                elif "2.bias" in k: mapped["fc2.bias"] = v
                elif ("4.weight" in k) and (v.shape == model.fc_out.weight.shape):
                    mapped["fc_out.weight"] = v
                elif ("4.bias" in k) and (v.shape == model.fc_out.bias.shape):
                    mapped["fc_out.bias"] = v
            missing, unexpected = model.load_state_dict({**model.state_dict(), **mapped}, strict=False)
            print(f"Transfer: missing={missing}, unexpected={unexpected}")
        except Exception as e:
            print(f"[warn] transfer-perc skipped ({e})")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score, best_state = -1.0, None
    global_thr = 0.5
    bce = nn.BCELoss()

    for stage in range(args.stages):
        # curriculum slice
        max_d = 1 + stage*(max(1,args.max_difficulty-1)//max(1,args.stages-1))
        tr_sub = curriculum_subset(tr, max_d)
        sampler = make_sampler(tr_sub, pos_weight=args.pos_sample_weight)
        dl_tr = DataLoader(tr_sub, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), collate_fn=collate)
        dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        beta = args.fn_beta_min + (args.fn_beta_max-args.fn_beta_min)*(stage/max(1,args.stages-1))
        margin = args.margin_min + (args.margin_max-args.margin_min)*(stage/max(1,args.stages-1))

        print(f"\n== Stage {stage+1}/{args.stages} (max_difficulty={max_d}) ==")
        epochs_this = max(1, args.epochs // args.stages)
        for ep in range(epochs_this):
            t0 = time.time()
            model.train(); run_loss = 0.0
            for X,y,_ in dl_tr:
                X = X.to(device); y = y.to(device)
                # Build [T,B,F]
                if args.rate_encode:
                    X_TBF = torch.stack([torch.bernoulli(X.clamp(0,1)) for _ in range(args.Ts)], dim=0)
                else:
                    X_TBF = X.unsqueeze(0).repeat(args.Ts, 1, 1)

                rate, prob, latency = model.forward_all(X_TBF)  # tensors [B]
                loss = bce(prob, y) \
                       + beta * fn_hinge(y, rate, margin) \
                       + sparsity(rate, lam=args.sparsity_lam) \
                       + latency_loss(latency, y, lam_pos=args.lat_pos, lam_neg=args.lat_neg)

                if args.focal_lambda > 0 and args.focal_gamma > 0:
                    loss = loss + args.focal_lambda * focal_binary(y, prob, gamma=args.focal_gamma)

                opt.zero_grad(); loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                run_loss += loss.item() * X.size(0)
            sched.step()

            # ---- validation ----
            model.eval()
            y_true_all, y_prob_all = [], []
            with torch.no_grad():
                for X,y,_ in dl_va:
                    X = X.to(device)
                    if args.rate_encode:
                        X_TBF = torch.stack([torch.bernoulli(X.clamp(0,1)) for _ in range(args.Ts)], dim=0)
                    else:
                        X_TBF = X.unsqueeze(0).repeat(args.Ts, 1, 1)
                    _, prob, _ = model.forward_all(X_TBF)
                    y_true_all.append(y.numpy()); y_prob_all.append(prob.cpu().numpy())

            y_true = np.concatenate(y_true_all); y_prob = np.concatenate(y_prob_all)
            # choose threshold favouring recall
            best_thr, best_f = global_thr, 0.0
            for thr in np.linspace(0.3, 0.8, 11):
                m = bin_metrics(y_true, y_prob, thr)
                score = 0.7*m["recall"] + 0.3*m["f1"]
                if score > best_f:
                    best_f, best_thr = score, thr
            global_thr = best_thr
            m = bin_metrics(y_true, y_prob, thr=global_thr); auc = roc_auc(y_true, y_prob)
            dt = time.time()-t0
            print(f"ep {ep+1}/{epochs_this}  loss {run_loss/max(1,len(dl_tr.dataset)):.4f}  "
                  f"val_f1 {m['f1']:.3f}  rec {m['recall']:.3f}  auc {auc:.3f}  thr {global_thr:.2f}  {dt:.1f}s")

            score = 0.7*m["recall"] + 0.3*m["f1"]
            if score > best_score:
                best_score = score
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, args.save)
        print(f"Saved best SNN weights to {args.save} (score={best_score:.3f})")
    else:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final SNN weights to {args.save}")

if __name__ == "__main__":
    main()
