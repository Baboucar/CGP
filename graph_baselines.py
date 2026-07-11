"""
Recent graph collaborative-filtering baselines, on the SAME backbone / data / eval
protocol as graphvit_model.py so the numbers drop straight into Table 1.

  NGCF      (SIGIR 2019)  plain graph CF: W1/W2 + LeakyReLU + bi-interaction, concat layers
  SGL-ED    (SIGIR 2021)  LightGCN + edge-dropout contrastive views (InfoNCE)
  DirectAU  (KDD   2022)  LightGCN/MF + alignment + uniformity (no negatives)
  LightGCL  (ICLR  2023)  LightGCN + truncated-SVD contrastive view (cosine InfoNCE)
  SCCF      (KDD   2024)  LightGCN encoder + unified contrastive kernel exp(s/t)+exp(s^2/t)

Data pipeline and 99-neg leave-one-out HR@10/NDCG@10 evaluation are imported verbatim
from graphvit_model.py so every method is measured identically.

  python graph_baselines.py --model ngcf     --data data/ml-1m --epochs 200 --reg 1e-5
  python graph_baselines.py --model sgl      --data data/ml-1m --epochs 90 --cl 0.02 --tau 0.2 --edge_drop 0.1
  python graph_baselines.py --model lightgcl --data data/ml-1m --epochs 120 --cl 0.005 --tau 0.2 --svd_q 5 --layers 2 --batch 4096
  python graph_baselines.py --model sccf     --data data/ml-1m --epochs 300 --tau 0.1  --batch 40000 --no_amp
  python graph_baselines.py --model directau --data data/ml-1m --epochs 120 --gamma 1  --layers 3
"""
import argparse, time, json
import numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from graphvit_model import (read_ratings, load_test, build_norm_adj, BPRData,
                            evaluate, bpr, info_nce, seed_all)


# ---------------------------------------------------------------- helpers
def build_edge_dropped_adj(tu, ti, n_users, n_items, keep_prob, device):
    """SGL-ED: keep each interaction edge w.p. keep_prob, then RE-normalize from surviving edges."""
    m = np.random.rand(len(tu)) < keep_prob
    su, si = tu[m], ti[m]
    N = n_users + n_items
    r = np.concatenate([su, si + n_users]); c = np.concatenate([si + n_users, su])
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, N))
    deg = np.asarray(A.sum(1)).flatten()
    dinv = np.power(deg, -0.5, where=deg > 0); dinv[deg == 0] = 0.0
    D = sp.diags(dinv)
    norm = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(device)


def build_rnorm(tu, ti, n_users, n_items, device):
    """Rectangular symmetric-normalized user->item block R_norm[u,i]=1/sqrt(deg_u*deg_i)."""
    R = sp.coo_matrix((np.ones(len(tu), np.float32), (tu, ti)), shape=(n_users, n_items))
    ru = np.asarray(R.sum(1)).flatten(); ci = np.asarray(R.sum(0)).flatten()
    du = np.power(ru, -0.5, where=ru > 0); du[ru == 0] = 0.0
    dc = np.power(ci, -0.5, where=ci > 0); dc[ci == 0] = 0.0
    Rn = (sp.diags(du) @ R @ sp.diags(dc)).tocoo()
    idx = torch.tensor(np.vstack([Rn.row, Rn.col]), dtype=torch.long)
    val = torch.tensor(Rn.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n_users, n_items)).coalesce().to(device)


def lightgcn_propagate(emb_weight, adj, n_layers, readout="mean"):
    with torch.autocast(device_type="cuda", enabled=False):
        e = emb_weight.float(); embs = [e]
        for _ in range(n_layers):
            e = torch.sparse.mm(adj, e); embs.append(e)
        stk = torch.stack(embs, 0)
        out = stk.mean(0) if readout == "mean" else stk.sum(0)
    return out


# ---------------------------------------------------------------- NGCF (2019)
class NGCF(nn.Module):
    def __init__(self, n_users, n_items, adj, d=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.register_buffer("adj", adj, persistent=False)
        self.emb = nn.Embedding(n_users + n_items, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self._cache = None

    def propagate(self):
        with torch.autocast(device_type="cuda", enabled=False):
            e = self.emb.weight.float()
            embs = [e]
            for l in range(self.n_layers):
                side = torch.sparse.mm(self.adj, e)          # neighbour aggregation  L e
                sum_part = self.W1[l](e + side)              # (L+I) e W1  (self-connection via +e)
                bi_part = self.W2[l](e * side)               # (L e ⊙ e) W2  bi-interaction
                e = self.drop(self.act(sum_part + bi_part))
                embs.append(F.normalize(e, p=2, dim=1))      # official NGCF L2-normalizes each propagated layer
            out = torch.cat(embs, dim=1)                     # concat layers (layer 0 raw) -> dim d*(L+1)
        return out[:self.n_users], out[self.n_users:]

    def refresh(self, perturb=False): self._cache = self.propagate()
    def score(self, u, i, perturb=False):
        eu, ei = self._cache; return (eu[u] * ei[i]).sum(-1)


# ---------------------------------------------------------------- SGL (2021) / SCCF (2024) / DirectAU (2022): LightGCN encoders
class LGNEncoder(nn.Module):
    """Plain LightGCN embeddings (mean readout incl. layer 0). Shared by SGL, SCCF, DirectAU."""
    def __init__(self, n_users, n_items, adj, d=64, n_layers=3, normalize_score=False):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.normalize_score = normalize_score
        self.register_buffer("adj", adj, persistent=False)
        self.emb = nn.Embedding(n_users + n_items, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self._cache = None

    def encode(self, adj=None):
        out = lightgcn_propagate(self.emb.weight, self.adj if adj is None else adj, self.n_layers, "mean")
        return out[:self.n_users], out[self.n_users:]

    def refresh(self, perturb=False):
        eu, ei = self.encode()
        if self.normalize_score:                 # SCCF ranks by cosine
            eu, ei = F.normalize(eu, dim=-1), F.normalize(ei, dim=-1)
        self._cache = (eu, ei)
    def score(self, u, i, perturb=False):
        eu, ei = self._cache; return (eu[u] * ei[i]).sum(-1)


# ---------------------------------------------------------------- LightGCL (2023)
class LightGCL(nn.Module):
    def __init__(self, n_users, n_items, rnorm, rnormT, u_mul_s, v_mul_s, ut, vt, d=64, n_layers=2):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.register_buffer("rnorm", rnorm, persistent=False)
        self.register_buffer("rnormT", rnormT, persistent=False)
        self.register_buffer("u_mul_s", u_mul_s, persistent=False)
        self.register_buffer("v_mul_s", v_mul_s, persistent=False)
        self.register_buffer("ut", ut, persistent=False)
        self.register_buffer("vt", vt, persistent=False)
        self.emb = nn.Embedding(n_users + n_items, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self._cache = None

    def propagate(self):
        """Summed main-view (E) and SVD-view (G); both read the MAIN view's previous layer."""
        with torch.autocast(device_type="cuda", enabled=False):
            eu = self.emb.weight[:self.n_users].float()
            ei = self.emb.weight[self.n_users:].float()
            Eu, Ei, Gu, Gi = [eu], [ei], [eu], [ei]
            for _ in range(self.n_layers):
                gu = self.u_mul_s @ (self.vt @ Ei[-1])
                gi = self.v_mul_s @ (self.ut @ Eu[-1])
                zu = torch.sparse.mm(self.rnorm, Ei[-1])
                zi = torch.sparse.mm(self.rnormT, Eu[-1])
                Eu.append(zu); Ei.append(zi); Gu.append(gu); Gi.append(gi)
            E_u = torch.stack(Eu, 0).sum(0); E_i = torch.stack(Ei, 0).sum(0)
            G_u = torch.stack(Gu, 0).sum(0); G_i = torch.stack(Gi, 0).sum(0)
        return E_u, E_i, G_u, G_i

    def refresh(self, perturb=False):
        E_u, E_i, _, _ = self.propagate(); self._cache = (E_u, E_i)
    def score(self, u, i, perturb=False):
        eu, ei = self._cache; return (eu[u] * ei[i]).sum(-1)


def lgcl_infonce(Gq, Eq, E_full, tau):
    """LightGCL contrastive (cosine — the paper-text form; raw-dot grows unboundedly with
    sum-readout embeddings and destabilises BPR), FULL node set as negatives."""
    Gq = F.normalize(Gq, dim=-1); Eq = F.normalize(Eq, dim=-1); E_full = F.normalize(E_full, dim=-1)
    pos = (Gq * Eq).sum(-1) / tau
    neg = torch.logsumexp((Gq @ E_full.t()) / tau, dim=1)
    return (-pos + neg).mean()


def sgl_infonce(q1, q2, all2, tau):
    """SGL-ED InfoNCE (cosine, FULL node set as negatives), one node type."""
    q1 = F.normalize(q1, dim=-1); q2 = F.normalize(q2, dim=-1); all2 = F.normalize(all2, dim=-1)
    pos = (q1 * q2).sum(-1) / tau
    logits = q1 @ all2.t() / tau
    return (-pos + torch.logsumexp(logits, dim=1)).mean()


def sccf_loss(U_all, V_all, bu, bi, tau, chunk=0):
    """SCCF unified kernel exp(s/t)+exp(s^2/t); in-batch cross product as negatives.
    The `down` term is chunked over users so the full user x item matrix is never materialised."""
    u = U_all[bu]; v = V_all[bi]
    ip = (u * v).sum(1)
    up = ((ip / tau).exp() + (ip.pow(2) / tau).exp()).log().mean()
    uu, cu = torch.unique(bu, return_counts=True)
    ii, ci = torch.unique(bi, return_counts=True)
    Uu = U_all[uu]; Vi = V_all[ii]; cu = cu.float(); ci = ci.float()
    nU, nI = uu.size(0), ii.size(0)
    if chunk and nU > chunk:
        tot = torch.zeros((), device=U_all.device)
        for s in range(0, nU, chunk):
            Sc = Uu[s:s+chunk] @ Vi.t()
            sc = (Sc / tau).exp() + (Sc.pow(2) / tau).exp()
            tot = tot + (sc * (cu[s:s+chunk].unsqueeze(1) * ci.unsqueeze(0))).sum()
        down = (tot / (nU * nI)).log()
    else:
        S = Uu @ Vi.t()
        score = (S / tau).exp() + (S.pow(2) / tau).exp()
        down = (score * (cu.unsqueeze(1) * ci.unsqueeze(0))).mean().log()
    return -up + down


def directau_loss(pu, qi, gamma=1.0, unif_max=1024):
    """DirectAU (2022): alignment + uniformity on L2-normalized embeddings.
    Uniformity estimated on a random subsample to keep the O(n^2) pdist cheap."""
    pu = F.normalize(pu, dim=-1); qi = F.normalize(qi, dim=-1)
    align = (pu - qi).norm(p=2, dim=1).pow(2).mean()
    def uniform(x):
        if x.size(0) > unif_max:
            x = x[torch.randperm(x.size(0), device=x.device)[:unif_max]]
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().add(1e-12).log()
    return align + gamma * (uniform(pu) + uniform(qi)) / 2


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ngcf", "sgl", "lightgcl", "sccf", "directau"])
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--cl", type=float, default=0.1)         # ssl weight (SGL, LightGCL)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)      # DirectAU uniformity weight
    ap.add_argument("--dropout", type=float, default=0.1)    # NGCF message dropout
    ap.add_argument("--edge_drop", type=float, default=0.1)  # SGL edge-drop ratio rho
    ap.add_argument("--svd_q", type=int, default=5)          # LightGCL SVD rank
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--num_negs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_chunk", type=int, default=256)
    ap.add_argument("--loss_chunk", type=int, default=0)     # SCCF: chunk the down-term over users
    ap.add_argument("--in_batch_neg", action="store_true")  # SGL/LightGCL: in-batch (not full-set) negatives
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save", default="")
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

    if a.model == "ngcf":
        model = NGCF(n_users, n_items, adj, d=a.d, n_layers=a.layers, dropout=a.dropout).to(dev)
    elif a.model == "sgl":
        model = LGNEncoder(n_users, n_items, adj, d=a.d, n_layers=a.layers).to(dev)
    elif a.model == "sccf":
        model = LGNEncoder(n_users, n_items, adj, d=a.d, n_layers=a.layers, normalize_score=True).to(dev)
    elif a.model == "directau":
        model = LGNEncoder(n_users, n_items, adj, d=a.d, n_layers=a.layers).to(dev)
    elif a.model == "lightgcl":
        rnorm = build_rnorm(tu, ti, n_users, n_items, dev)
        rnormT = rnorm.t().coalesce()
        U, S, V = torch.svd_lowrank(rnorm, q=a.svd_q, niter=7)
        u_mul_s = (U * S).contiguous(); v_mul_s = (V * S).contiguous()
        ut = U.t().contiguous(); vt = V.t().contiguous()
        model = LightGCL(n_users, n_items, rnorm, rnormT, u_mul_s, v_mul_s, ut, vt, d=a.d, n_layers=a.layers).to(dev)

    opt = (torch.optim.SGD(model.parameters(), lr=a.lr) if a.opt == "sgd"
           else torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd))
    amp = (dev.type == "cuda") and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    nP = sum(p.numel() for p in model.parameters())
    print(f"[{a.tag or a.model}] users={n_users} items={n_items} params={nP/1e6:.3f}M "
          f"model={a.model} layers={a.layers} cl={a.cl} tau={a.tau} gamma={a.gamma} lr={a.lr} batch={a.batch}", flush=True)

    best_hr = best_ndcg = -1; best_ep = -1; hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = tb = tc = 0.0; nb = 0
        adj1 = adj2 = None
        if a.model == "sgl" and a.cl > 0:
            adj1 = build_edge_dropped_adj(tu, ti, n_users, n_items, 1 - a.edge_drop, dev)
            adj2 = build_edge_dropped_adj(tu, ti, n_users, n_items, 1 - a.edge_drop, dev)
        for u, pos, negs in dl:
            u = u.to(dev, non_blocking=True); pos = pos.to(dev, non_blocking=True)
            negs = negs.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                lc = torch.tensor(0.0, device=dev)
                if a.model == "ngcf":
                    eu, ei = model.propagate()
                    nf = negs.reshape(-1)
                    ps = (eu[u] * ei[pos]).sum(-1); ns = (eu[u] * ei[nf]).sum(-1)
                    lb = bpr(ps, ns)
                    e0 = model.emb.weight
                    reg = (e0[u].pow(2).sum(-1).mean() + e0[pos].pow(2).sum(-1).mean()
                           + e0[nf].pow(2).sum(-1).mean()) * 0.5
                    loss = lb + a.reg * reg

                elif a.model == "sgl":
                    eu, ei = model.encode(model.adj)
                    nf = negs.reshape(-1)
                    ps = (eu[u] * ei[pos]).sum(-1); ns = (eu[u] * ei[nf]).sum(-1)
                    lb = bpr(ps, ns)
                    if a.cl > 0:
                        v1 = lightgcn_propagate(model.emb.weight, adj1, model.n_layers, "mean")
                        v2 = lightgcn_propagate(model.emb.weight, adj2, model.n_layers, "mean")
                        ub = torch.unique(u); ib = torch.unique(pos) + model.n_users
                        nU = model.n_users
                        with torch.autocast(device_type="cuda", enabled=False):
                            v1f, v2f = v1.float(), v2.float()
                            if a.in_batch_neg:
                                lc = (info_nce(v1f[ub], v2f[ub], a.tau) + info_nce(v1f[ib], v2f[ib], a.tau))
                            else:
                                lc = (sgl_infonce(v1f[ub], v2f[ub], v2f[:nU], a.tau)
                                      + sgl_infonce(v1f[ib], v2f[ib], v2f[nU:], a.tau))
                    e0 = model.emb.weight
                    reg = (e0[u].pow(2).sum(-1).mean() + e0[pos].pow(2).sum(-1).mean()
                           + e0[nf].pow(2).sum(-1).mean()) * 0.5
                    loss = lb + a.cl * lc + a.reg * reg

                elif a.model == "lightgcl":
                    E_u, E_i, G_u, G_i = model.propagate()
                    nf = negs.reshape(-1)
                    ps = (E_u[u] * E_i[pos]).sum(-1); ns = (E_u[u] * E_i[nf]).sum(-1)
                    lb = bpr(ps, ns)
                    iid = torch.cat([pos, nf])
                    with torch.autocast(device_type="cuda", enabled=False):
                        E_uf, E_if = E_u.float(), E_i.float(); G_uf, G_if = G_u.float(), G_i.float()
                        if a.in_batch_neg:
                            lc = (lgcl_infonce(G_uf[u], E_uf[u], E_uf[u], a.tau)
                                  + lgcl_infonce(G_if[iid], E_if[iid], E_if[iid], a.tau))
                        else:
                            lc = (lgcl_infonce(G_uf[u], E_uf[u], E_uf, a.tau)
                                  + lgcl_infonce(G_if[iid], E_if[iid], E_if, a.tau))
                    e0 = model.emb.weight
                    reg = (e0[u].pow(2).sum(-1).mean() + e0[pos].pow(2).sum(-1).mean()
                           + e0[nf].pow(2).sum(-1).mean()) * 0.5
                    loss = lb + a.cl * lc + a.reg * reg

                elif a.model == "sccf":
                    with torch.autocast(device_type="cuda", enabled=False):
                        eu, ei = model.encode()
                        U_all = F.normalize(eu.float(), dim=-1); V_all = F.normalize(ei.float(), dim=-1)
                        loss = sccf_loss(U_all, V_all, u, pos, a.tau, chunk=a.loss_chunk)
                    lb = loss

                elif a.model == "directau":
                    with torch.autocast(device_type="cuda", enabled=False):
                        eu, ei = model.encode()
                        loss = directau_loss(eu[u].float(), ei[pos].float(), a.gamma)
                        if a.reg > 0:
                            e0 = model.emb.weight
                            loss = loss + a.reg * 0.5 * (e0[u].pow(2).sum(-1).mean() + e0[pos].pow(2).sum(-1).mean())
                    lb = loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); tb += float(lb); tc += float(lc); nb += 1
        line = f"ep {ep:03d} | loss={tot/nb:.4f} main={tb/nb:.4f} cl={tc/nb:.4f} | {time.time()-t0:.0f}s"
        if ep % a.eval_every == 0 or ep == a.epochs:
            hr, nd = evaluate(model, cands, dev, k=10, chunk=a.eval_chunk)
            hist.append(dict(epoch=ep, hr=hr, ndcg=nd))
            flag = ""
            if hr > best_hr:
                best_hr, best_ndcg, best_ep = hr, nd, ep; flag = "  *best"
                if a.save:
                    torch.save(model.state_dict(), a.save)
                    json.dump({"args": vars(a), "best_hr": best_hr, "best_ndcg": best_ndcg, "best_ep": best_ep,
                               "params": nP, "hist": hist}, open(a.save + ".json", "w"), indent=2)  # persist each best (robust to sleep-hangs)
            line += f" | HR@10={hr:.4f} NDCG@10={nd:.4f}{flag}"
        print(line, flush=True)
    print(f"BEST [{a.tag or a.model}] HR@10={best_hr:.4f} NDCG@10={best_ndcg:.4f} @ep{best_ep}", flush=True)
    if a.save:
        json.dump({"args": vars(a), "best_hr": best_hr, "best_ndcg": best_ndcg, "best_ep": best_ep,
                   "params": nP, "hist": hist}, open(a.save + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
