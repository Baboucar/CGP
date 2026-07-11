"""
Graph-powered ViT for collaborative filtering (LightGViT).

  LightGCN propagation  ->  strong graph embeddings  (base score = dot product)
       + ViT interaction encoder on the outer product  (correction term)
       + XSimGCL-style contrastive regulariser on the propagated embeddings

Legit target on ML-1M (leave-one-out, 99 neg): HR@10 ~0.72-0.75.

  python graphvit.py --model lgn       --epochs 120          # LightGCN backbone only
  python graphvit.py --model lgn_vit   --epochs 120 --cl 0.1 # full graph-powered ViT
"""
import argparse, re, time, json, math
from pathlib import Path
import numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seed_all(s=42):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ------------------------------------------------------------------ data
def read_ratings(path):
    us, it = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, i = line.split("\t")[:2]
            us.append(int(u)); it.append(int(i))
    return np.array(us, np.int64), np.array(it, np.int64)

_pair = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
def load_test(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = _pair.match(line); u, pos = int(m.group(1)), int(m.group(2))
            negs = [int(x) for x in re.findall(r"\d+", line[m.end():])]
            out[u] = (pos, negs)
    return out

def build_norm_adj(tu, ti, n_users, n_items, device):
    """Symmetric normalized adjacency of the bipartite user-item graph."""
    N = n_users + n_items
    r = np.concatenate([tu, ti + n_users]); c = np.concatenate([ti + n_users, tu])
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, N))
    deg = np.asarray(A.sum(1)).flatten()
    dinv = np.power(deg, -0.5, where=deg > 0); dinv[deg == 0] = 0.0
    D = sp.diags(dinv)
    norm = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(device)

class BPRData(Dataset):
    def __init__(self, tu, ti, n_items, num_negs=1):
        self.u = tu; self.i = ti; self.n_items = n_items; self.num_negs = num_negs
        self.pos = set(zip(tu.tolist(), ti.tolist()))
    def __len__(self): return len(self.u)
    def __getitem__(self, k):
        u = int(self.u[k]); pos = int(self.i[k]); negs = []
        while len(negs) < self.num_negs:
            j = np.random.randint(self.n_items)
            if (u, j) not in self.pos: negs.append(j)
        return torch.tensor(u), torch.tensor(pos), torch.tensor(negs)

# ------------------------------------------------------------------ model
class TransformerBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h, d), nn.Dropout(dropout))
    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + a; return x + self.mlp(self.n2(x))

class LightGViT(nn.Module):
    def __init__(self, n_users, n_items, adj, d=64, n_layers=3, use_vit=True,
                 patch=16, depth=2, heads=8, mlp_ratio=2.0, dropout=0.1,
                 alpha=0.5, eps=0.1, cl_layer=1):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers, self.use_vit, self.alpha, self.eps, self.cl_layer = n_layers, use_vit, alpha, eps, cl_layer
        self.register_buffer("adj", adj, persistent=False)
        self.emb = nn.Embedding(n_users + n_items, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        if use_vit:
            assert d % patch == 0
            self.inst = nn.InstanceNorm2d(1, affine=True)
            self.proj = nn.Conv2d(1, d, patch, patch)
            npatch = (d // patch) ** 2
            self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
            self.pos = nn.Parameter(torch.zeros(1, npatch + 1, d)); nn.init.normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList([TransformerBlock(d, heads, mlp_ratio, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # ViT term starts at 0
        self._cache = None

    def propagate(self, perturb=False):
        # sparse.mm has no fp16 kernel -> force fp32 for the graph convolution
        with torch.autocast(device_type="cuda", enabled=False):
            e = self.emb.weight.float()
            embs = [e]; cl_view = None
            for layer in range(self.n_layers):
                e = torch.sparse.mm(self.adj, e)
                if perturb:
                    noise = F.normalize(torch.rand_like(e), dim=-1) * torch.sign(e) * self.eps
                    e = e + noise
                embs.append(e)
                if layer == self.cl_layer:
                    cl_view = e
            out = torch.stack(embs, 0).mean(0)
            eu, ei = out[:self.n_users], out[self.n_users:]
            if cl_view is None: cl_view = out
        return eu, ei, cl_view

    def vit_term(self, pu, qi):
        M = torch.bmm(pu.unsqueeze(2), qi.unsqueeze(1)).unsqueeze(1)
        M = self.inst(M)
        x = self.proj(M).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x)[:, 0]).squeeze(-1)

    def refresh(self, perturb=False):          # cache propagated embeddings for eval
        self._cache = self.propagate(perturb)[:2]
    def score(self, u, i, perturb=False):      # used by evaluate()
        eu, ei = self._cache
        pu, qi = eu[u], ei[i]
        s = (pu * qi).sum(-1)
        if self.use_vit: s = s + self.alpha * self.vit_term(pu, qi)
        return s

# ------------------------------------------------------------------ loss / eval
def bpr(pos, neg): return -F.logsigmoid(pos - neg).mean()
def info_nce(a, b, tau=0.2):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)   # L2 along feature dim
    return F.cross_entropy(a @ b.t() / tau, torch.arange(a.size(0), device=a.device))

@torch.no_grad()
def evaluate(model, cands, device, k=10, chunk=256):
    model.eval(); model.refresh(perturb=False)
    users = sorted(cands); C = 1 + len(cands[users[0]][1])
    HR = np.zeros(len(users)); ND = np.zeros(len(users))
    for a in range(0, len(users), chunk):
        blk = users[a:a+chunk]; ur, ir = [], []
        for u in blk:
            pos, negs = cands[u]
            ur.append(np.full(1+len(negs), u, np.int64)); ir.append(np.asarray([pos]+negs, np.int64))
        u_t = torch.from_numpy(np.concatenate(ur)).to(device)
        i_t = torch.from_numpy(np.concatenate(ir)).to(device)
        s = model.score(u_t, i_t).float().view(len(blk), C)
        gt = (s[:, 1:] > s[:, :1]).sum(1).float(); eq = (s[:, 1:] == s[:, :1]).sum(1).float()
        rank = (gt + 0.5*eq).cpu().numpy(); hit = rank < k
        HR[a:a+len(blk)] = hit; ND[a:a+len(blk)] = np.where(hit, 1/np.log2(rank+2), 0.0)
    return HR.mean(), ND.mean()

# ------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgn_vit", choices=["lgn", "lgn_vit"])
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--reg", type=float, default=1e-4)   # batch-wise L2 on ego embeddings
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--num_negs", type=int, default=1)
    ap.add_argument("--cl", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--cl_layer", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--init", default="")     # warm-start embeddings from a checkpoint
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seed_all(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tu, ti = read_ratings(a.data + ".train.rating")
    cands = load_test(a.data + ".test.negative")
    n_users = max(int(tu.max()) + 1, max(cands) + 1)
    n_items = int(ti.max()) + 1
    for _, ng in cands.values(): n_items = max(n_items, max(ng) + 1)
    adj = build_norm_adj(tu, ti, n_users, n_items, dev)
    ds = BPRData(tu, ti, n_items, a.num_negs)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)

    model = LightGViT(n_users, n_items, adj, d=a.d, n_layers=a.layers,
                      use_vit=(a.model == "lgn_vit"), patch=a.patch, depth=a.depth,
                      heads=a.heads, dropout=a.dropout, alpha=a.alpha, eps=a.eps, cl_layer=a.cl_layer).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        loaded = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {a.init}: missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    amp = (dev.type == "cuda") and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    nP = sum(p.numel() for p in model.parameters())
    print(f"[{a.tag or a.model}] users={n_users} items={n_items} params={nP/1e6:.3f}M "
          f"layers={a.layers} vit={a.model=='lgn_vit'} alpha={a.alpha} cl={a.cl} negs={a.num_negs} lr={a.lr}", flush=True)

    best_hr = best_ndcg = -1; best_ep = -1; hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = tb = tc = 0.0; nb = 0
        for u, pos, negs in dl:
            u = u.to(dev, non_blocking=True); pos = pos.to(dev, non_blocking=True); negs = negs.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                eu, ei, v1 = model.propagate(perturb=False)
                model._cache = (eu, ei)
                pu = eu[u]; pi = ei[pos]
                ps = (pu * pi).sum(-1)
                ur = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
                nf = negs.reshape(-1)
                ns = (eu[ur] * ei[nf]).sum(-1)
                if model.use_vit:
                    ps = ps + model.alpha * model.vit_term(pu, pi)
                    ns = ns + model.alpha * model.vit_term(eu[ur], ei[nf])
                ns = ns.view(u.size(0), -1)
                lb = bpr(ps.unsqueeze(1).expand_as(ns).reshape(-1), ns.reshape(-1))
                loss = lb; lc = torch.tensor(0.0, device=dev)
                if a.reg > 0:      # batch-wise L2 on layer-0 (ego) embeddings, LightGCN-style
                    e0 = model.emb.weight
                    reg = (e0[u].pow(2).sum(-1).mean() + e0[pos].pow(2).sum(-1).mean()
                           + e0[nf].pow(2).sum(-1).mean()) * 0.5
                    loss = loss + a.reg * reg
                if a.cl > 0:
                    _, _, v2 = model.propagate(perturb=True)
                    nodes = torch.cat([u, pos + model.n_users]).unique()
                    with torch.autocast(device_type="cuda", enabled=False):   # fp32: avoids fp16 normalize -> NaN
                        lc = info_nce(v1[nodes].float(), v2[nodes].float(), a.tau)
                    loss = loss + a.cl * lc
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); tb += lb.item(); tc += float(lc); nb += 1
        line = f"ep {ep:03d} | loss={tot/nb:.4f} bpr={tb/nb:.4f} cl={tc/nb:.4f} | {time.time()-t0:.0f}s"
        if ep % a.eval_every == 0 or ep == a.epochs:
            hr, nd = evaluate(model, cands, dev, k=10)
            hist.append(dict(epoch=ep, hr=hr, ndcg=nd))
            flag = ""
            if hr > best_hr:
                best_hr, best_ndcg, best_ep = hr, nd, ep; flag = "  *best"
                if a.save:
                    torch.save(model.state_dict(), a.save)
                    json.dump({"args": vars(a), "best_hr": best_hr, "best_ndcg": best_ndcg, "best_ep": best_ep,
                               "params": nP, "hist": hist}, open(a.save + ".json", "w"), indent=2)  # persist each best
            line += f" | HR@10={hr:.4f} NDCG@10={nd:.4f}{flag}"
        print(line, flush=True)
    print(f"BEST [{a.tag or a.model}] HR@10={best_hr:.4f} NDCG@10={best_ndcg:.4f} @ep{best_ep}", flush=True)
    if a.save:
        json.dump({"args": vars(a), "best_hr": best_hr, "best_ndcg": best_ndcg, "best_ep": best_ep,
                   "params": nP, "hist": hist}, open(a.save + ".json", "w"), indent=2)

if __name__ == "__main__":
    main()
