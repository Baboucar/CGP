"""
Clean experiment harness for ViT-based collaborative filtering (Vit4Rec) + baselines.

Fixes vs. the original notebook:
  * Correct top-K evaluation: rank the positive by counting how many of the 99
    negatives score strictly higher. No reliance on argsort tie order (the bug
    that produced HR=NDCG=1.0000 for an untrained model).
  * Model trained with raw-logit BPR (no premature sigmoid), BatchNorm on the
    interaction map, learnable positional + CLS token, proper init -> it actually
    learns (BPR loss drops well below ln 2 = 0.693).
  * Optional XSimGCL-style contrastive regulariser.

Usage:
  python rec.py --model vitrec --epochs 30
  python rec.py --model neumf  --epochs 30
"""
import argparse, os, random, re, time, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------- data
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_rating_file(path):
    us, it = [], []
    mu, mi = -1, -1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, i = line.split("\t")[:2]
            u, i = int(u), int(i)
            us.append(u); it.append(i)
            mu = max(mu, u); mi = max(mi, i)
    return np.asarray(us, np.int64), np.asarray(it, np.int64), mu + 1, mi + 1

_pair = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
def load_test_candidates(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = _pair.match(line)
            u, pos = int(m.group(1)), int(m.group(2))
            rest = line[m.end():]
            negs = [int(x) for x in re.findall(r"\d+", rest)]
            out[u] = (pos, negs)
    return out

class BPRData(Dataset):
    def __init__(self, coo, n_items, num_negs=1):
        self.rows = coo.row.astype(np.int64)
        self.cols = coo.col.astype(np.int64)
        self.n_items = n_items
        self.num_negs = num_negs
        self.pos_set = set(zip(self.rows.tolist(), self.cols.tolist()))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        u = int(self.rows[idx]); pos = int(self.cols[idx])
        negs = []
        while len(negs) < self.num_negs:
            j = np.random.randint(self.n_items)
            if (u, j) not in self.pos_set: negs.append(j)
        return (torch.tensor(u), torch.tensor(pos),
                torch.tensor(negs, dtype=torch.long))

# ----------------------------------------------------------------------------- models
class TransformerBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, d), nn.Dropout(dropout))
    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x

class VitRec(nn.Module):
    """Outer-product interaction map -> ViT -> BPR score (raw logit).
    Optional GMF fusion branch keeps the strong bilinear match signal available."""
    def __init__(self, n_users, n_items, d=64, patch=16, depth=2, heads=8,
                 mlp_ratio=2.0, dropout=0.1, use_cls=True, use_pos=True, eps=0.1,
                 map_norm="instance", fuse_gmf=False, sep_emb=False):
        super().__init__()
        self.d, self.eps = d, eps
        self.fuse_gmf = fuse_gmf
        self.sep_emb = sep_emb and fuse_gmf
        self.P = nn.Embedding(n_users, d)
        self.Q = nn.Embedding(n_items, d)
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        if self.sep_emb:                       # separate embedding table for the GMF branch
            self.Pg = nn.Embedding(n_users, d); self.Qg = nn.Embedding(n_items, d)
            nn.init.normal_(self.Pg.weight, std=0.01); nn.init.normal_(self.Qg.weight, std=0.01)
        assert d % patch == 0
        # per-sample map normalisation avoids the batch-statistic leakage a
        # BatchNorm would cause when positives/negatives are scored separately.
        self.map_norm = map_norm
        self.inst = nn.InstanceNorm2d(1, affine=True) if map_norm == "instance" else None
        self.proj = nn.Conv2d(1, d, kernel_size=patch, stride=patch)
        n_patch = (d // patch) ** 2
        self.use_cls = use_cls
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
        n_tok = n_patch + (1 if use_cls else 0)
        self.use_pos = use_pos
        if use_pos:
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        head_in = d * 2 if fuse_gmf else d
        self.head = nn.Sequential(nn.Linear(head_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def _perturb(self, e):
        n = F.normalize(torch.rand_like(e), dim=-1)
        return e + torch.sign(e) * n * self.eps

    def encode(self, u_ids, i_ids, perturb=False):
        u = self.P(u_ids); v = self.Q(i_ids)
        if perturb:
            u = self._perturb(u); v = self._perturb(v)
        M = torch.bmm(u.unsqueeze(2), v.unsqueeze(1)).unsqueeze(1)  # (B,1,d,d)
        if self.map_norm == "instance":
            M = self.inst(M)
        elif self.map_norm == "l2":                                # unit-norm embeddings
            M = torch.bmm(F.normalize(u, dim=-1).unsqueeze(2),
                          F.normalize(v, dim=-1).unsqueeze(1)).unsqueeze(1)
        x = self.proj(M).flatten(2).transpose(1, 2)                # (B, n_patch, d)
        if self.use_cls:
            x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        if self.use_pos:
            x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x[:, 0] if self.use_cls else x.mean(1)
        if self.fuse_gmf:
            g = self.Pg(u_ids) * self.Qg(i_ids) if self.sep_emb else u * v
            feat = torch.cat([feat, g], dim=-1)
        return feat

    def score(self, u_ids, i_ids, perturb=False):
        return self.head(self.encode(u_ids, i_ids, perturb)).squeeze(-1)
    def forward(self, u_ids, i_ids, perturb=False):
        f = self.encode(u_ids, i_ids, perturb)
        return f, self.head(f).squeeze(-1)

class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, d=64, **kw):
        super().__init__()
        self.P = nn.Embedding(n_users, d); self.Q = nn.Embedding(n_items, d)
        nn.init.normal_(self.P.weight, std=0.01); nn.init.normal_(self.Q.weight, std=0.01)
    def score(self, u, i, perturb=False):
        return (self.P(u) * self.Q(i)).sum(-1)

class GMF(nn.Module):
    def __init__(self, n_users, n_items, d=64, **kw):
        super().__init__()
        self.P = nn.Embedding(n_users, d); self.Q = nn.Embedding(n_items, d)
        self.h = nn.Linear(d, 1)
        nn.init.normal_(self.P.weight, std=0.01); nn.init.normal_(self.Q.weight, std=0.01)
    def score(self, u, i, perturb=False):
        return self.h(self.P(u) * self.Q(i)).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, n_users, n_items, d=64, dropout=0.1, **kw):
        super().__init__()
        self.P = nn.Embedding(n_users, d); self.Q = nn.Embedding(n_items, d)
        self.net = nn.Sequential(nn.Linear(2*d, 2*d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(2*d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d//2), nn.ReLU())
        self.h = nn.Linear(d//2, 1)
        nn.init.normal_(self.P.weight, std=0.01); nn.init.normal_(self.Q.weight, std=0.01)
    def score(self, u, i, perturb=False):
        x = torch.cat([self.P(u), self.Q(i)], -1)
        return self.h(self.net(x)).squeeze(-1)

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, d=64, dropout=0.1, **kw):
        super().__init__()
        self.Pg = nn.Embedding(n_users, d); self.Qg = nn.Embedding(n_items, d)
        self.Pm = nn.Embedding(n_users, d); self.Qm = nn.Embedding(n_items, d)
        self.net = nn.Sequential(nn.Linear(2*d, 2*d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(2*d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d//2), nn.ReLU())
        self.h = nn.Linear(d + d//2, 1)
        for e in [self.Pg, self.Qg, self.Pm, self.Qm]:
            nn.init.normal_(e.weight, std=0.01)
    def score(self, u, i, perturb=False):
        g = self.Pg(u) * self.Qg(i)
        m = self.net(torch.cat([self.Pm(u), self.Qm(i)], -1))
        return self.h(torch.cat([g, m], -1)).squeeze(-1)

class ConvNCF(nn.Module):
    """He et al. 2018: outer-product interaction map -> CNN tower -> BPR score.
    The direct CNN counterpart of our ViT encoder (same input map)."""
    def __init__(self, n_users, n_items, d=64, channels=32, dropout=0.0, **kw):
        super().__init__()
        self.P = nn.Embedding(n_users, d); self.Q = nn.Embedding(n_items, d)
        nn.init.normal_(self.P.weight, std=0.1); nn.init.normal_(self.Q.weight, std=0.1)
        n_layers = int(round(np.log2(d)))          # 64 -> 6 halving conv layers -> 1x1
        layers, c_in, sp = [], 1, d
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c_in, channels, kernel_size=2, stride=2))
            sp //= 2
            if sp > 1:                              # per-sample norm (spatial>1); no batch leakage
                layers.append(nn.InstanceNorm2d(channels, affine=True))
            layers.append(nn.ReLU())
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)
    def score(self, u, i, perturb=False):
        M = torch.bmm(self.P(u).unsqueeze(2), self.Q(i).unsqueeze(1)).unsqueeze(1)  # (B,1,d,d)
        x = self.conv(M).flatten(1)                # (B, channels)
        return self.head(x).squeeze(-1)

MODELS = {"vitrec": VitRec, "bprmf": BPRMF, "gmf": GMF, "mlp": MLP,
          "neumf": NeuMF, "convncf": ConvNCF}

# ----------------------------------------------------------------------------- loss / eval
def bpr_loss(pos, neg):
    return -F.logsigmoid(pos - neg).mean()

def info_nce(a, b, tau=0.2):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    logits = a @ b.t() / tau
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)

@torch.no_grad()
def evaluate(model, cands, device, k=10, user_chunk=256):
    """Vectorised leave-one-out eval. Positive is column 0 of each candidate row;
    rank = number of negatives scoring strictly higher than the positive."""
    model.eval()
    users = sorted(cands.keys())
    C = 1 + len(cands[users[0]][1])               # candidates per user (100)
    HR = np.zeros(len(users)); NDCG = np.zeros(len(users))
    for a in range(0, len(users), user_chunk):
        blk = users[a:a + user_chunk]
        u_rows, i_rows = [], []
        for u in blk:
            pos, negs = cands[u]
            u_rows.append(np.full(1 + len(negs), u, np.int64))
            i_rows.append(np.asarray([pos] + negs, np.int64))
        u_t = torch.from_numpy(np.concatenate(u_rows)).to(device)
        i_t = torch.from_numpy(np.concatenate(i_rows)).to(device)
        s = model.score(u_t, i_t).float().view(len(blk), C)
        pos_s = s[:, :1]
        gt = (s[:, 1:] > pos_s).sum(1).float()
        eq = (s[:, 1:] == pos_s).sum(1).float()
        rank = (gt + 0.5 * eq).cpu().numpy()   # average-rank tie handling
        hit = rank < k
        HR[a:a + len(blk)] = hit
        NDCG[a:a + len(blk)] = np.where(hit, 1.0 / np.log2(rank + 2), 0.0)
    return HR.mean(), NDCG.mean()

# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitrec", choices=list(MODELS))
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num_negs", type=int, default=1)
    ap.add_argument("--lambda_cl", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--no_cls", action="store_true")
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--map_norm", default="instance", choices=["none", "instance", "l2"])
    ap.add_argument("--fuse_gmf", action="store_true")
    ap.add_argument("--sep_emb", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_chunk", type=int, default=256)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_u, tr_i, nu1, ni1 = read_rating_file(args.data + ".train.rating")
    cands = load_test_candidates(args.data + ".test.negative")
    n_users = max(nu1, max(cands) + 1)
    n_items = ni1
    for _, negs in cands.values():
        n_items = max(n_items, max(negs) + 1)
    coo = sp.coo_matrix((np.ones(len(tr_u), np.float32), (tr_u, tr_i)), shape=(n_users, n_items))
    ds = BPRData(coo, n_items, num_negs=args.num_negs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    mk = MODELS[args.model]
    model = mk(n_users, n_items, d=args.d, patch=args.patch, depth=args.depth,
               heads=args.heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
               use_cls=not args.no_cls, use_pos=not args.no_pos, eps=args.eps,
               map_norm=args.map_norm, fuse_gmf=args.fuse_gmf, sep_emb=args.sep_emb).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"[{args.tag or args.model}] users={n_users} items={n_items} params={n_params/1e6:.3f}M "
          f"d={args.d} patch={args.patch} lr={args.lr} bs={args.batch} negs={args.num_negs} "
          f"cl={args.lambda_cl} cls={not args.no_cls} pos={not args.no_pos}", flush=True)

    best_hr, best_ndcg, best_ep = -1, -1, -1
    hist = []
    use_cl = args.lambda_cl > 0 and args.model == "vitrec"
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        tot = tot_bpr = tot_cl = 0.0; nb = 0
        for u, pos, negs in dl:
            u = u.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            negs = negs.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                if use_cl:
                    f1, ps = model(u, pos, perturb=False)
                    f2, _ = model(u, pos, perturb=True)
                else:
                    ps = model.score(u, pos)
                ur = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
                ns = model.score(ur, negs.reshape(-1)).view(u.size(0), -1)
                lb = bpr_loss(ps.unsqueeze(1).expand_as(ns).reshape(-1), ns.reshape(-1))
                loss = lb
                lc = torch.tensor(0.0)
                if use_cl:
                    lc = info_nce(f1, f2, tau=args.tau)
                    loss = loss + args.lambda_cl * lc
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); tot_bpr += lb.item(); tot_cl += float(lc); nb += 1
        sched.step()
        hr, ndcg = evaluate(model, cands, device, k=args.k, user_chunk=args.eval_chunk)
        hist.append(dict(epoch=ep, loss=tot/nb, bpr=tot_bpr/nb, cl=tot_cl/nb, hr=hr, ndcg=ndcg))
        flag = ""
        if hr > best_hr:
            best_hr, best_ndcg, best_ep = hr, ndcg, ep; flag = "  *best"
            if args.save:
                torch.save(model.state_dict(), args.save)
        print(f"ep {ep:02d} | loss={tot/nb:.4f} bpr={tot_bpr/nb:.4f} cl={tot_cl/nb:.4f} "
              f"| HR@{args.k}={hr:.4f} NDCG@{args.k}={ndcg:.4f} | {time.time()-t0:.0f}s{flag}", flush=True)
    print(f"BEST [{args.tag or args.model}] HR@{args.k}={best_hr:.4f} NDCG@{args.k}={best_ndcg:.4f} @ep{best_ep}", flush=True)
    if args.save:
        json.dump({"args": vars(args), "hist": hist, "best_hr": best_hr,
                   "best_ndcg": best_ndcg, "best_ep": best_ep, "params": n_params},
                  open(args.save + ".json", "w"), indent=2)

if __name__ == "__main__":
    main()
