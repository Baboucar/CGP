# model.py  (embedding-dimension now fully parametrised)
from minimal_transformer import SimpleVisionTransformer
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ViT-based CF model   —   outer-product → vision transformer → score.
    `embed_dim` can be freely changed (16, 32, 64, …) without shape errors.
    """

    def __init__(self, user_count: int, item_count: int, embed_dim: int = 64):
        super().__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.embed_dim  = embed_dim          # << single source of truth

        # ---------- ID embeddings ----------
        self.P = nn.Embedding(user_count, embed_dim)
        self.Q = nn.Embedding(item_count, embed_dim)

        # ---------- Vision Transformer backbone ----------
        self.vit = SimpleVisionTransformer(
            in_chans=3,                 # we replicate 1→3 channels
            embed_dim=embed_dim,        # must match embed_dim
            patch_size=8,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # ---------- prediction head ----------
        self.fc1      = nn.Linear(embed_dim, embed_dim)      # keeps width = d
        self.dropout  = nn.Dropout(0.2)
        self.out      = nn.Linear(user_count, 1)             # final score

        # convenience for view()
        self.spatial_shape = [embed_dim, embed_dim]

    # --------------------------------------------------
    def forward(self, user_ids, item_ids):
        """
        user_ids / item_ids : 1-D LongTensor [batch]
        returns             : 1-D FloatTensor [batch] (sigmoid scores)
        """
        # 1) embeddings
        u = self.P(user_ids)             # (B, d)
        v = self.Q(item_ids)             # (B, d)

        # 2) outer product → interaction “image”
        map_ = torch.bmm(u.unsqueeze(2), v.unsqueeze(1))      # (B,d,d)
        map_ = map_.view(-1, 1, *self.spatial_shape)          # (B,1,d,d)
        img  = map_.repeat(1, 3, 1, 1)                        # (B,3,d,d)

        # 3) ViT backbone
        x = self.vit(img)                                     # (B,d)
        x = self.dropout(torch.relu(x))

        # 4) head
        x = self.fc1(x)                                       # (B,d)
        x = x @ self.P.weight.T                               # (B,user_count)
        x = self.dropout(x)
        logits = self.out(x).squeeze(1)                       # (B,)

        return torch.sigmoid(logits)
