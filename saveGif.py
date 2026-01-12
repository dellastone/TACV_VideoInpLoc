import os
import re
import torch
import numpy as np
import imageio.v2 as imageio
import argparse
from pathlib import Path
from tqdm import tqdm

import utils.datasets as datasets
from iml_vit_model import IMLViT
from torchvision.io import read_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _denorm_imagenet(img_chw: torch.Tensor) -> torch.Tensor:
    """img_chw: [3,H,W] float tensor"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3, 1, 1)
    img = img_chw * std + mean
    return img.clamp(0, 1)


def _frame_sort_key(frame_name: str):
    """
    Tries to sort frame names numerically if they contain a number, otherwise lexicographically.
    Works for e.g. '000123.png', 'frame_12.jpg', etc.
    """
    m = re.search(r"(\d+)", frame_name)
    return (0, int(m.group(1))) if m else (1, frame_name)


def _render_4panel_to_rgb(
    img_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    pred_prob_chw: torch.Tensor,
    orig_img_chw: torch.Tensor,
    orig_h: int,
    orig_w: int,
    pred_thr: float = 0.5,
) -> np.ndarray:
    """
    Returns an RGB uint8 image (H,W,3) of the 4-panel figure.
    """
    img = img_chw[:, :orig_h, :orig_w].detach().cpu()
    gt  = gt_chw[:, :orig_h, :orig_w].detach().cpu()
    pr  = pred_prob_chw[:, :orig_h, :orig_w].detach().cpu()

    pred_bin = (pr > pred_thr).float()

    gt3 = gt.repeat(3, 1, 1)
    pb3 = pred_bin.repeat(3, 1, 1)

    heat = pr.squeeze(0).clamp(0, 1)  # [H,W]

    alpha = 0.35  # transparency strength (0 = invisible, 1 = solid)
 
    alpha_map = alpha * heat

    overlay = orig_img_chw[:, :orig_h, :orig_w].detach().cpu().clone()
    red = torch.zeros_like(overlay)
    red[0] = 1.0  # pure red overlay

    # alpha blend: overlay = (1-a)*img + a*red
    overlay = overlay * (1 - alpha_map.unsqueeze(0)) + red * alpha_map.unsqueeze(0)
    
    overlay = overlay.clamp(0, 1)

    imgs = [img, gt3, pb3, overlay]
    labels = ["Inpainted image", "GT mask", "Predicted mask", "Real image + overlay"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for ax, im, lab in zip(axes, imgs, labels):
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.text(0.5, -0.08, lab, transform=ax.transAxes, ha="center", va="top", fontsize=12)

    plt.subplots_adjust(bottom=0.22, wspace=0.02)

    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()

    plt.close(fig)
    return rgb


@torch.no_grad()
def save_video_visualization_gif(
    model,
    loader,
    device,
    video_id: str,
    out_gif_path: str,
    orig_frames_root: str = "/media/NAS/AINPAINT/DATASET_AInpaint/input_frames/432x240_recompressed_h264",
    orig_h: int = 240,
    orig_w: int = 432,
    pred_thr: float = 0.5,
    fps: float = 5.0,
    max_frames: int | None = None,
):
    """
    Iterates over `loader`, selects samples whose meta['video_id'] == `video_id`,
    renders the same 4-panel visualization as save_visualization, and writes to a GIF.
    """
    os.makedirs(os.path.dirname(out_gif_path) or ".", exist_ok=True)


    collected = []

    model.eval()

    for x, masks, edge_masks, meta in loader:
        x = x.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edge_masks = edge_masks.to(device, non_blocking=True)


        B = x.shape[0]
        for b in range(B):
            if meta["video_id"][b] != video_id:
                continue
            _, mask_pred_prob, _ = model(x, masks, edge_masks)
            
            collected.append({
                "x": x[b].detach().cpu(),
                "mask": masks[b].detach().cpu(),
                "pred": mask_pred_prob[b].detach().cpu(),
                "video_id": meta["video_id"][b],
                "frame_name": meta["frame_name"][b],
            })

    if len(collected) == 0:
        print(f"[gif] No frames found for video_id='{video_id}' in this loader.")
        return

    collected.sort(key=lambda d: _frame_sort_key(d["frame_name"]))

    duration = 1.0 / max(fps, 1e-6)
    with imageio.get_writer(out_gif_path, mode="I", duration=duration) as writer:
        for i, item in enumerate(collected):
            if max_frames is not None and i >= max_frames:
                break

            # Read the real/original image for overlay
            orig_img_path = os.path.join(orig_frames_root, item["video_id"], item["frame_name"])
            orig_img = read_image(orig_img_path).float() / 255.0  # [3,H,W] in [0,1]

            # Denorm the inpainted network input for visualization
            img_vis = _denorm_imagenet(item["x"])

            frame_rgb = _render_4panel_to_rgb(
                img_chw=img_vis,
                gt_chw=item["mask"],             # [1,H,W]
                pred_prob_chw=item["pred"],      # [1,H,W]
                orig_img_chw=orig_img,           # [3,H,W]
                orig_h=orig_h,
                orig_w=orig_w,
                pred_thr=pred_thr,
            )
            writer.append_data(frame_rgb)

    print(f"[gif] Saved GIF for video_id='{video_id}' with {min(len(collected), max_frames or len(collected))} frames to: {out_gif_path}")


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
def main():
    args = parse_args()
    device = get_device(args.device)

    root_dir = args.root_dir
    csv_path = args.csv_path or str(Path(root_dir) / "dataset.csv")

    # Build loaders (we only use test_loader, but this matches your pipeline)
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
    save_video_visualization_gif(
        model=model,
        loader=test_loader,
        device=device,
        video_id="airplane_6_bis",  
        out_gif_path="./val_vis/video_vis_dinov2_airplane.gif",
        pred_thr=0.5,
        fps=5,
    )

    


if __name__ == "__main__":
    main()
