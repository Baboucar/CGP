from minimal_transformer import SimpleVisionTransformer
import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, user_count, item_count):
        super(Model, self).__init__()

        self.user_count = user_count
        self.item_count = item_count

        # Embedding settings
        self.embedding_size = 64
        self.spatial_shape = [64, 64]
        self.flat_shape = [10, 10]
        self.k_size = [3, 3]
        self.stride = 1
        self.filtered = [(self.flat_shape[0] - self.k_size[0]) // self.stride + 1,
                         (self.flat_shape[1] - self.k_size[1]) // self.stride + 1]

        fc_length = self.filtered[0] * self.filtered[1]
        self.fc1 = torch.nn.Linear(64, fc_length)

        # User/Item Embeddings
        self.P = nn.Embedding(self.user_count, self.embedding_size).cuda()
        self.Q = nn.Embedding(self.item_count, self.embedding_size).cuda()

        # ----------------------------------------------------------------------
        # REPLACE CNN WITH A SIMPLE VISION TRANSFORMER
        # ----------------------------------------------------------------------
        # Instead of:
        #   self.in_layer = nn.Conv2d(3, self.channel_size*2, self.kernel_size, ...)
        #   ...
        # We define a small ViT:
        self.vit = SimpleVisionTransformer(
            in_chans=3,  # because we concat 3 copies of m below
            embed_dim=64,  # must match your desired feature size
            patch_size=8,  # choose patch size carefully
            depth=2,  # number of TransformerEncoderBlocks
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1
        )

        self.in_drop = nn.Dropout(0.2)
        self.in_relu = nn.ReLU()

        self.out_drp = nn.Dropout(0.2)
        self.out_layer = nn.Linear(self.user_count, 1)

    def forward(self, user_ids, item_ids):
        # Convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        # 1) Get user/item embeddings
        user_embeddings = self.P(torch.tensor(user_ids).cuda())  # [B, 64]
        item_embeddings = self.Q(torch.tensor(item_ids).cuda())  # [B, 64]

        # 2) Construct the 2D interaction map
        #    user_2d: [B, 64, 1], item_2d: [B, 1, 64]
        user_2d = user_embeddings.unsqueeze(2)
        item_2d = item_embeddings.unsqueeze(1)

        # Outer product => shape [B, 64, 64] -> we add channel dimension = 1
        m = torch.bmm(user_2d, item_2d)  # [B, 64, 64]
        m = m.view(-1, 1, *self.spatial_shape)  # [B, 1, 64, 64]

        # For the ViT, we can replicate the channel 3x to mimic RGB
        # or you can keep it 1 and adjust the ViT's in_chans=1
        p = torch.cat([m, m, m], dim=1)  # [B, 3, 64, 64]

        # ----------------------------------------------------------------------
        # Pass the 2D map through the Vision Transformer
        # ----------------------------------------------------------------------
        x = self.vit(p)  # [B, embed_dim] = [B, 64]

        x = self.in_drop(x)
        x = self.in_relu(x)  # [B, 64]

        # 3) Now, you have a 1D representation from the ViT (x).
        #    If you still want to use your fc1 or other operations, do so here.
        #    For example, your original code had:
        #      x = x.view(256, -1, [2, 2]) ...
        #    But that shape logic was for your CNN. So let's adapt carefully.

        # Example: Apply fc1 to a 4D input? We might not do that anymore.
        # Instead, let's do something simpler to keep the pipeline:
        x = self.fc1(x)  # [B, fc_length], but fc_length= filtered[0]*filtered[1]

        # 4) Multiply by user embedding matrix transpose
        x = torch.mm(x, self.P.weight.transpose(1, 0))  # [B, user_count]

        x = self.out_drp(x)
        x = self.out_layer(x)  # [B, 1]

        pred = torch.sigmoid(x)
        return pred.view(-1)
