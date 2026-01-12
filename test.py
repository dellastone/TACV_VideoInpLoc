import argparse
from pathlib import Path
import torch
from tqdm import tqdm

import utils.datasets as datasets
from iml_vit_model import IMLViT
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
def make_valid_mask_like(masks, orig_h=240, orig_w=432):
    # masks: [B,1,H,W]
    valid = torch.zeros_like(masks, dtype=torch.bool)
    valid[..., :orig_h, :orig_w] = True
    return valid

def parse_args():
    p = argparse.ArgumentParser("Test IMLViT checkpoint on test set")

    # data
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root directory of the dataset (contains dataset.csv)")
    p.add_argument("--csv_path", type=str, default=None,
                   help="Path to dataset.csv (default: <root_dir>/dataset.csv)")
    p.add_argument("--method", type=str, required=True,
                   help="Single method to evaluate (e.g., STTN, OPN)")

    # checkpoint
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to .pth checkpoint saved by save_checkpoint()")

    # model config (must match training)
    p.add_argument("--input_size", type=int, default=432)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--encoder_type", type=str, default="dinov2",
                   choices=["window_vit", "dinov2", "mae", "dinov3"])
    p.add_argument("--vit_pretrain_path", type=str, default="./mae_pretrain_vit_base.pth",
                   help="Only used if encoder_type=window_vit")
    p.add_argument("--use_fpn", action="store_true", default=False)
    p.add_argument("--fpn_channels", type=int, default=256)
    p.add_argument("--mlp_emb_dim", type=int, default=256)
    p.add_argument("--predict_head_norm", type=str, default="BN",
                   choices=["BN", "LN", "IN", "none"])

    # eval settings
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])

    return p.parse_args()


def get_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args, device):
    model = IMLViT(
        input_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_type=args.encoder_type,
        vit_pretrain_path=args.vit_pretrain_path if args.encoder_type == "window_vit" else None,
        use_fpn=args.use_fpn,
        fpn_channels=args.fpn_channels,
        mlp_embeding_dim=args.mlp_emb_dim,
        predict_head_norm=args.predict_head_norm,
        edge_lambda=0.0,
    ).to(device)
    return model

def save_visualization(x, masks, mask_pred_prob, meta):
  # pick sample index in the batch
    vis_dir = "./val_vis"
    os.makedirs(vis_dir, exist_ok=True)
    b = 0
    orig_img_path = os.path.join("/media/NAS/AINPAINT/DATASET_AInpaint/input_frames/432x240", meta['video_id'][b], meta['frame_name'][b])
    orig_img = read_image(orig_img_path).float() / 255.0  # [3,H,W], in [0,1]
    # x is normalized most likely; for visualization clamp to [0,1]
    img = x[b].detach().cpu()
    #remove imagenet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img * std + mean

    img = img.clamp(0, 1)

    gt = masks[b].detach().cpu()          # [1,H,W]
    pred = mask_pred_prob[b].detach().cpu()  # [1,H,W]

    pred_bin = (pred > 0.5).float()

    # make 3-channel masks so they can be stacked with the image
    gt3 = gt.repeat(3, 1, 1)
    pred_bin3 = pred_bin.repeat(3, 1, 1)
    orig_h = 240
    orig_w = 432
    img = img[:, :orig_h, :orig_w]
    gt3 = gt3[:, :orig_h, :orig_w]
    pred_bin3 = pred_bin3[:, :orig_h, :orig_w]
    pred = pred[:, :orig_h, :orig_w]
    # optional overlay: red = prediction
    overlay = orig_img.clone()
    
    overlay[0] = torch.maximum(overlay[0], pred.squeeze(0))  # boost red channel where predicted
    overlay = overlay.clamp(0, 1)
    # Ensure everything is on CPU
    imgs = [img.cpu(), gt3.cpu(), pred_bin3.cpu(), overlay.cpu()]
    labels = ["Inpainted image", "GT mask", "Predicted mask", "Real image + overlay"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, im, lab in zip(axes, imgs, labels):
        # CHW -> HWC for imshow
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.axis("off")
        # caption UNDER the image
        ax.text(
            0.5, -0.08, lab,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=12
        )

    # leave room at the bottom for captions
    plt.subplots_adjust(bottom=0.22, wspace=0.02)

    out_path = os.path.join(vis_dir, "val_last_batch_vis.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluate] Saved visualization to: {out_path}")
    
@torch.no_grad()
def evaluate_micro_f1(model, loader, device, thresholds):
    model.eval()
    tp = {t: 0.0 for t in thresholds}
    fp = {t: 0.0 for t in thresholds}
    fn = {t: 0.0 for t in thresholds}
    vis_dir = "./val_vis"
    os.makedirs(vis_dir, exist_ok=True)
    for x, masks, edge_masks, meta in tqdm(loader, desc="Testing", leave=False):
        x = x.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edge_masks = edge_masks.to(device, non_blocking=True)

        _, mask_pred_prob, _ = model(x, masks, edge_masks)

        valid = make_valid_mask_like(masks, 240, 432)       # <-- NEW
        target = (masks > 0.5) & valid                      # <-- CHANGED

        for t in thresholds:
            preds = (mask_pred_prob > t) & valid            # <-- CHANGED
            tp[t] += (preds & target).sum().item()
            fp[t] += (preds & ~target).sum().item()
            fn[t] += ((~preds) & target).sum().item()
        # Save visualization for the last batch only   
        save_visualization(x, masks, mask_pred_prob, meta)

    eps = 1e-7
    f1s = {t: (2 * tp[t]) / (2 * tp[t] + fp[t] + fn[t] + eps) for t in thresholds}
    best_t = max(f1s, key=f1s.get)
    return f1s, best_t, f1s[best_t]

@torch.no_grad()
def evaluate_macro_f1(model, loader, device, thresholds, empty_score=1.0):
    model.eval()
    f1_sum = {t: 0.0 for t in thresholds}
    n_imgs = 0

    for x, masks, edge_masks, meta in tqdm(loader, desc="Testing (macro-F1)", leave=False):
        x = x.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edge_masks = edge_masks.to(device, non_blocking=True)

        _, mask_pred_prob, _ = model(x, masks, edge_masks)

        valid = make_valid_mask_like(masks, 240, 432)      # <-- NEW
        target = (masks > 0.5) & valid                     # <-- CHANGED

        B = masks.shape[0]
        n_imgs += B

        for t in thresholds:
            preds = (mask_pred_prob > t) & valid           # <-- CHANGED

            tp = (preds & target).flatten(1).sum(1).float()
            fp = (preds & ~target).flatten(1).sum(1).float()
            fn = (~preds & target).flatten(1).sum(1).float()

            denom = 2 * tp + fp + fn
            f1 = torch.where(
                denom > 0,
                (2 * tp) / (denom + 1e-7),
                torch.full_like(denom, float(empty_score))
            )
            f1_sum[t] += f1.sum().item()

    f1s = {t: f1_sum[t] / max(n_imgs, 1) for t in thresholds}
    best_t = max(f1s, key=f1s.get)
    return f1s, best_t, f1s[best_t]


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Your save_checkpoint stores state in "model_state"
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        # fallback if you saved raw state_dict
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[WARN] Missing keys ({len(missing)}): e.g. {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): e.g. {unexpected[:10]}")


def main():
    args = parse_args()
    device = get_device(args.device)

    root_dir = args.root_dir
    csv_path = args.csv_path or str(Path(root_dir) / "dataset.csv")

    # Build loaders (we only use test_loader)
    test_loader = datasets.build_test_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        methods=[args.method],          # evaluate only one method
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args, device)
    load_checkpoint(model, args.ckpt_path, device)

    thresholds = tuple(float(t) for t in args.thresholds)
    macrof1s, macro_best_t, macro_best_f1 = evaluate_macro_f1(model, test_loader, device, thresholds)
    
    print("\n=== TEST RESULTS ===")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Method:     {args.method}")
    for t in thresholds:
        print(f"Macro-F1@{t:.2f}: {macrof1s[t]:.6f}")
    print(f"Best:       F1={macro_best_f1:.6f} @thr={macro_best_t:.2f}")    
    f1s, best_t, best_f1 = evaluate_micro_f1(model, test_loader, device, thresholds)

    for t in thresholds:
        print(f"Micro-F1@{t:.2f}: {f1s[t]:.6f}")
    print(f"Best:       F1={best_f1:.6f} @ thr={best_t:.2f}")
    

    


if __name__ == "__main__":
    main()
