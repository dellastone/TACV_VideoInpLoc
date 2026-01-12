from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.window_attention_ViT import ViT as window_attention_vit, SimpleFeaturePyramid, LastLevelMaxPool
from modules.decoderhead import PredictHead  
import utils.datasets as datasets 

class IMLViT(nn.Module):
    def __init__(
        self,
        # input / encoder config
        input_size: int = 1024, patch_size: int = 16, embed_dim: int = 768,

        # encoder choice
        encoder_type: str = "window_vit",   # "window_vit", "dino", "mae"
        vit_pretrain_path: str = None,      # used for window_vit (MAE/BEiT weights etc.)

        # SFPN config
        use_fpn: bool = True, fpn_channels: int = 256, fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),

        # decoder head
        mlp_embeding_dim: int = 256, predict_head_norm: str = "BN",

        # edge loss
        edge_lambda: float = 20.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.encoder_type = encoder_type
        self.use_fpn = use_fpn
        self.edge_lambda = edge_lambda
        self.vit_pretrain_path = vit_pretrain_path

        # Build encoder
        self.encoder_net, encoder_out_dim = self._build_encoder(
            encoder_type=encoder_type,
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            vit_pretrain_path=vit_pretrain_path,
        )

        # Optional SFPN
        if use_fpn:
            if patch_size == 14:
                fpn_scale_factors = (2.0, 1.0, 0.5, 0.25)
            self.featurePyramid_net = SimpleFeaturePyramid(input_dim=encoder_out_dim, out_channels=fpn_channels, input_stride=patch_size, scale_factors=fpn_scale_factors, top_block=LastLevelMaxPool(), norm="LN")
            head_in_channels = [fpn_channels for i in range(5)]    # SFPN returns 5 scales
        else:
            self.featurePyramid_net = None          # no FPN, single feature map
            head_in_channels = [encoder_out_dim]

        # Decoder head
        self.predict_head = PredictHead(
            feature_channels=head_in_channels,
            embed_dim=mlp_embeding_dim,
            norm=predict_head_norm,
        )


        self.BCE_loss = nn.BCEWithLogitsLoss()


        # Weight init
        if self.featurePyramid_net is not None:
            self.featurePyramid_net.apply(self._init_weights)
        self.predict_head.apply(self._init_weights)

    def dice_loss_from_logits(self,logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        dice = (2*inter + eps) / (union + eps)
        return 1 - dice.mean()

    def _build_encoder( self, encoder_type, img_size, patch_size, embed_dim, vit_pretrain_path ):
        """
        Build encoder and return (encoder_module, out_channels).
        """

        if encoder_type == "window_vit":
            enc = window_attention_vit(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=12,
                num_heads=12,
                drop_path_rate=0.1,
                window_size=14,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                residual_block_indexes=[],
                use_rel_pos=True,
                out_feature="last_feat",
            )
            # optional pretrained weights (e.g. MAE/BEiT)
            if vit_pretrain_path is not None:
                state = torch.load(vit_pretrain_path, map_location="cpu")
                enc.load_state_dict(state["model"], strict=False)
                print(f"[window_vit] Loaded pretrained weights from '{vit_pretrain_path}'")
            out_dim = embed_dim
            return enc, out_dim

        elif encoder_type in ["dinov2", "mae", "dinov3"]:
            import timm

            if encoder_type == "dinov2":
                model_name = "vit_base_patch14_dinov2.lvd142m"
            elif encoder_type == "dinov3":
                model_name = 'vit_base_patch16_dinov3.lvd1689m'
            else:  # "mae"
                model_name = "vit_base_patch16_224.mae"

            backbone = timm.create_model(
                model_name,
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
            )

            # Wrap timm ViT so that forward(x) -> (B, C, H', W')
            encoder = ViTBackboneWrapper(backbone)
            out_dim = backbone.embed_dim
            return encoder, out_dim

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x: torch.Tensor, masks: torch.Tensor, edge_masks: torch.Tensor, shape=None):
        """
        Args:
            x:          [B, 3, H, W]
            masks:      [B, 1, H, W] ground-truth mask
            edge_masks: [B, 1, H, W] edge weights
        Returns:
            total_loss, mask_pred_prob, edge_loss
        """
        # encoder output: [B, C, H', W']
        enc_out = self._encode(x)

        # collect feature maps for the head
        if self.use_fpn:
            fpn_dict = self.featurePyramid_net({"last_feat": enc_out})
            feature_list = list(fpn_dict.values())   # 5 levels
        else:
            feature_list = [enc_out]                 # single scale

        # decoder head
        logits = self.predict_head(feature_list)     # [B, 1, H', W']

        # upsample to input_size
        mask_pred = F.interpolate(logits, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        # main BCE
        predict_loss = self.BCE_loss(mask_pred, masks)
        dice = self.dice_loss_from_logits(mask_pred, masks)
        # edge-aware loss
        edge_loss = F.binary_cross_entropy_with_logits(input=mask_pred, target=masks, weight=edge_masks) * self.edge_lambda

        total_loss = predict_loss +  dice 
        mask_pred_prob = torch.sigmoid(mask_pred)

        return total_loss, mask_pred_prob, edge_loss
    
    def freeze_encoder(self):
        for p in self.encoder_net.parameters():
            p.requires_grad = False
        self.encoder_net.eval()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run encoder and return a 4D feature map [B, C, H', W'].
        """
        out = self.encoder_net(x)

        # window_vit: returns dict with 'last_feat'
        if isinstance(out, dict) and "last_feat" in out:
            return out["last_feat"]  # [B, C, H', W']

        # wrapped ViT / other encoders already return [B, C, H', W']
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            return out

        raise RuntimeError("Unsupported encoder output format in _encode().")


class ViTBackboneWrapper(nn.Module):
    """
    Wrap timm ViT so forward(x) returns [B, C, H, W].
    """

    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.embed_dim = vit_model.embed_dim

    def forward(self, x):
        out = self.vit.forward_features(x)

        if isinstance(out, dict):
            if "x" in out:
                tokens = out["x"]
            elif "last_hidden_state" in out:
                tokens = out["last_hidden_state"]
            else:
                raise RuntimeError("Unknown dict structure from vit.forward_features")
        else:
            tokens = out  # assume [B, N, C]

        # drop cls token if present
        if tokens.dim() == 3 and tokens.shape[1] > 1:
            n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            patch_tokens = tokens[:, n_prefix:, :]
        else:
            patch_tokens = tokens

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)   # Square root of N
        assert H * W == N, "Number of patches is not a perfect square, can't reshape into grid."

        # convert 1D token sequence into a 2D feature map
        feat = patch_tokens.transpose(1, 2).reshape(B, C, H, W)    # (B,N,C) --> (B,C,N) and then it flattens N into the 2D grid (H x W)

        return feat


if __name__ == "__main__":
    from pathlib import Path
    import torch
    from torch.optim import Adam
    from tqdm import tqdm
    root_dir = "/media/NAS/AINPAINT/DATASET_AInpaint"   # <-- adjust
    csv_path = str(Path(root_dir) / "dataset.csv")

    # dataloaders
    train_loader, val_loader, test_loader = datasets.build_all_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        methods=["STTN", "OPN"],          
        input_size=240,        
        batch_size=64,
        num_workers=4,
    )
    
    #print length of train_loader, val_loader, test_loader
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    print(f"Test loader length: {len(test_loader)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IMLViT(
        input_size=240,
        patch_size=16,
        embed_dim=768,
        encoder_type="window_vit",           # or "dino", "mae"
        vit_pretrain_path="./mae_pretrain_vit_base.pth",
        use_fpn=True,
        fpn_channels=256,
        mlp_embeding_dim=256,
        predict_head_norm="BN",
        edge_lambda=0.0,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()
    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        x, masks, edge_masks, meta = batch  # meta is a dict of lists

        x = x.to(device)
        masks = masks.to(device)
        edge_masks = edge_masks.to(device)

        optimizer.zero_grad()
        total_loss, mask_pred_prob, edge_loss = model(x, masks, edge_masks)
        total_loss.backward()
        optimizer.step()

        # Update tqdm postfix
        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            edge_loss=f"{edge_loss.item():.4f}"
        )

