import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for (B, C, H, W).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PredictHead(nn.Module):
    """
    Generalized decoder head:
      - accepts list of feature maps with possibly different channels / spatial sizes
      - projects all to `embed_dim`, resizes to same spatial size, concatenates, then predicts.
    """
    def __init__(
        self, 
        feature_channels: list,
        embed_dim: int = 256,
        predict_channels: int = 1,
        norm: str = "BN"
    ) -> None:
        super().__init__()
        assert len(feature_channels) >= 1, "feature_channels must have at least one element"
        self.use_proj = not all(c == embed_dim for c in feature_channels)

        self.num_feats = len(feature_channels)

        # 1x1 conv to unify channels -> embed_dim
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(c_in, embed_dim, kernel_size=1)
            for c_in in feature_channels
        ])

        # Fuse all projected features
        self.linear_fuse = nn.Conv2d(
            in_channels=embed_dim * self.num_feats,
            out_channels=embed_dim,
            kernel_size=1
        )

        assert norm in ["LN", "BN", "IN"], \
            "norm must be one of 'LN', 'BN', 'IN'"

        if norm == "LN":
            self.norm = LayerNorm2d(embed_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(embed_dim)
        else:  # "IN"
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout()
        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def forward(self, feats):
        """
        feats: list[Tensor], each (B, C_i, H_i, W_i)
        """
        assert isinstance(feats, (list, tuple)), "PredictHead expects a list/tuple of feature maps"
        assert len(feats) == self.num_feats

        # use the first feature's size as the target
        target_h, target_w = feats[0].shape[-2:]
        proj_resized = []

        for i, f in enumerate(feats):
            if self.use_proj:
                f = self.proj_convs[i](f)
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            proj_resized.append(f)

        x = torch.cat(proj_resized, dim=1)   # (B, embed_dim*num_feats, H, W)
        x = self.linear_fuse(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear_predict(x)
        return x
