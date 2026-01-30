# train_temporal_pole_full.py
import os, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Aeternum.temporal_pole_emotion_classifier import TemporalPoleEmotionClassifier

class PooledDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = np.load(X_path).astype(np.float32)   # [N,H]
        self.Y = np.load(Y_path).astype(np.float32)   # [N,4]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def collate(batch):
    X = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    Y = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    return X, Y

@torch.no_grad()
def eval_mae(model, dl, device):
    model.head.eval()
    mae = []
    for X,Y in dl:
        X = X.to(device); Y = Y.to(device)
        pred = model.head(X)      # [B,4]
        mae.append(torch.abs(pred - Y).mean(dim=0).cpu().numpy())
    mae = np.mean(np.stack(mae, axis=0), axis=0)
    return dict(valence=mae[0], arousal=mae[1], dominance=mae[2], surprise=mae[3], mean=float(mae.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-X", required=True)
    ap.add_argument("--train-Y", required=True)
    ap.add_argument("--val-X", required=True)
    ap.add_argument("--val-Y", required=True)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="temporal_pole_head.pt")
    args = ap.parse_args()

    tr = PooledDataset(args.train_X, args.train_Y)
    va = PooledDataset(args.val_X, args.val_Y)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = args.device
    head = TemporalPoleEmotionClassifier(hidden_dim=args.hidden, device=device)
    params = list(head.head.parameters())
    opt = optim.Adam(params, lr=args.lr)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=max(1,len(dl_tr)), epochs=args.epochs)
    loss = nn.L1Loss()

    best_mean = 1e9
    for ep in range(args.epochs):
        head.head.train()
        t0 = time.time()
        running = 0.0
        for X,Y in dl_tr:
            X = X.to(device); Y = Y.to(device)
            pred = head.head(X)
            l = loss(pred, Y)
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            running += l.item() * X.size(0)
        train_mae = running / max(1,len(dl_tr.dataset))
        val = eval_mae(head, dl_va, device)
        dt = time.time()-t0
        print(f"ep {ep+1}/{args.epochs} train_mae {train_mae:.4f} val_mae {val['mean']:.4f} (V {val['valence']:.3f} A {val['arousal']:.3f} D {val['dominance']:.3f} S {val['surprise']:.3f})  {dt:.1f}s")
        if val["mean"] < best_mean:
            best_mean = val["mean"]
            torch.save(head.head.state_dict(), args.save)
            print(f"  saved best to {args.save}")

if __name__ == "__main__":
    main()
