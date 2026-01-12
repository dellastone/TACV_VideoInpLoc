# main_train.py
import argparse
from pathlib import Path
import os

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

import utils.datasets as datasets
from iml_vit_model import IMLViT
import wandb
import torch



def build_optimizer_param_groups(model: nn.Module, head_lr: float, enc_lr: float, weight_decay: float = 0.01):
    """
    Creates AdamW with 2 param groups:
      - head/FPN params at head_lr
      - encoder params at enc_lr (only those with requires_grad=True)
    """
    head_params = []
    enc_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder_net"):
            enc_params.append(p)
        else:
            head_params.append(p)

    optimizer = AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": enc_params, "lr": enc_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer, head_params, enc_params


def freeze_all_vit(model: nn.Module):
    vit = getattr(model.encoder_net, "vit", None)
    if vit is None:
        raise RuntimeError("Expected model.encoder_net.vit (timm ViT wrapped by ViTBackboneWrapper)")
    for p in vit.parameters():
        p.requires_grad = False


def unfreeze_vit(model: nn.Module, n_blocks: int, unfreeze_norm: bool = True):
    """
    n_blocks = 0   -> keep encoder fully frozen
    n_blocks > 0   -> unfreeze last n transformer blocks
    n_blocks = -1  -> unfreeze all blocks (full end-to-end)
    """
    vit = getattr(model.encoder_net, "vit", None)
    if vit is None or not hasattr(vit, "blocks"):
        raise RuntimeError("This helper expects a timm ViT with `.blocks`")

    freeze_all_vit(model)

    if n_blocks == -1:
        for p in vit.parameters():
            p.requires_grad = True
    elif n_blocks > 0:
        for blk in vit.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True

    if unfreeze_norm:
        for norm_name in ["norm", "fc_norm"]:
            if hasattr(vit, norm_name) and getattr(vit, norm_name) is not None:
                for p in getattr(vit, norm_name).parameters():
                    p.requires_grad = True


def build_optimizer_param_groups(model: nn.Module, head_lr: float, enc_lr: float, weight_decay: float = 0.01):
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder_net"):
            enc_params.append(p)
        else:
            head_params.append(p)

    # If encoder is fully frozen, enc_params can be empty; that’s OK.
    param_groups = [{"params": head_params, "lr": head_lr}]
    if len(enc_params) > 0:
        param_groups.append({"params": enc_params, "lr": enc_lr})

    opt = AdamW(param_groups, weight_decay=weight_decay)
    return opt, head_params, enc_params


def binary_f1(pred_probs, target, threshold=0.5, eps=1e-7):
    preds = (pred_probs > threshold).float()
    target = (target > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    tp = (preds * target).sum(dim=1)
    fp = (preds * (1 - target)).sum(dim=1)
    fn = ((1 - preds) * target).sum(dim=1)

    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return f1.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train IMLViT for different inpainting methods")

    # Dataset / paths
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the dataset (contains dataset.csv)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to dataset.csv (default: <root_dir>/dataset.csv)")

    # Methods (STTN, OPN, GMCNN, ...)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["STTN", "OPN"],
        help='List of methods, e.g. --methods STTN OPN (must match your dataset spec)'
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Model / encoder config
    parser.add_argument("--input_size", type=int, default=432,
                        help="Input size (H=W) fed to the model & dataloader")
    parser.add_argument("--patch_size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--embed_dim", type=int, default=768, help="ViT embedding dimension")
    parser.add_argument("--encoder_type", type=str, default="window_vit",
                        choices=["window_vit", "dinov2", "mae", "dinov3"], help="Encoder type")

    parser.add_argument("--vit_pretrain_path", type=str, default="./mae_pretrain_vit_base.pth",
                        help="Path to ViT pretrained weights (only used for window_vit)")

    # SFPN / head
    parser.add_argument("--use_fpn", action="store_true", default=False, help="Use Simple Feature Pyramid")
    parser.add_argument("--fpn_channels", type=int, default=256, help="Number of channels in FPN output")
    parser.add_argument("--mlp_emb_dim", type=int, default=256, help="Decoder MLP embedding dim")
    parser.add_argument("--predict_head_norm", type=str, default="BN",
                        choices=["BN", "LN", "IN", "none"], help="Norm type in PredictHead")

    # Misc
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help='Device to use: "cuda", "cpu", or "mps". Default: auto')

    return parser.parse_args()


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(args):
    root_dir = args.root_dir
    csv_path = args.csv_path or str(Path(root_dir) / "dataset.csv")

    train_loader, val_loader, test_loader = datasets.build_all_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        methods=args.methods,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length:   {len(val_loader)}")
    print(f"Test loader length:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def build_model(args, device):
    print("use fpn:", args.use_fpn)
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
        edge_lambda=0.0,  # edge loss lambda fixed to 0
    ).to(device)

    return model


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device):
    """
    MACRO-F1 over the validation set:
      - compute F1 per image
      - average across images
    """
    model.eval()

    thresholds = (0.1, 0.3, 0.5)

    total_loss = 0.0
    total_batches = 0

    # accumulate sum of per-image F1 over entire val set (macro-F1)
    f1_sum = {t: 0.0 for t in thresholds}
    n_imgs = 0

    pbar = tqdm(val_loader, desc="Validation")
    for x, masks, edge_masks, meta in pbar:
        x = x.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edge_masks = edge_masks.to(device, non_blocking=True)

        loss, mask_pred_prob, edge_loss = model(x, masks, edge_masks)
        total_loss += loss.item()
        total_batches += 1

        # binarize target once
        target = (masks > 0.5)
        B = masks.size(0)
        n_imgs += B

        for t in thresholds:
            preds = (mask_pred_prob > t)

            # per-image TP/FP/FN across pixels
            tp = (preds & target).flatten(1).sum(dim=1).float()      # [B]
            fp = (preds & ~target).flatten(1).sum(dim=1).float()     # [B]
            fn = ((~preds) & target).flatten(1).sum(dim=1).float()   # [B]

            denom = 2 * tp + fp + fn
            eps = 1e-7

            f1 = torch.where(
                denom > 0,
                (2 * tp) / (denom + eps),
                torch.ones_like(denom)
            )

            f1_sum[t] += f1.sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_f1s = {t: (f1_sum[t] / max(n_imgs, 1)) for t in thresholds}

    best_t = max(val_f1s, key=val_f1s.get)
    best_val_f1 = val_f1s[best_t]

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, val_f1s, best_t, best_val_f1


def save_checkpoint(model, optimizer, epoch, best_val_loss, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    # ckpt_path = os.path.join(save_dir, f"{name}_checkpoint_epoch_{epoch:03d}.pth")
    ckpt_path = os.path.join(save_dir, f"{name}_checkpoint.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    name = f"IMLViT_bs{args.batch_size}_enc{args.encoder_type}_embed{args.embed_dim}_padding_432_fpn{args.use_fpn}_ImagenetNorm_epochs{args.epochs}_noDiceLoss"
    wandb.init(project="IMLViT_Training", config=vars(args), name=name)

    train_loader, val_loader, test_loader = build_dataloaders(args)
    model = build_model(args, device)

 
    stages = [
        ("end2end", args.epochs, -1, 1e-5, 5e-6, 0.05),
    ]
    patience = 3
    no_improve_epochs = 0
    min_delta = 1e-4

    best_f1 = 0.0
    global_epoch = 0

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for stage_name, stage_epochs, unfreeze_blocks, head_lr, enc_lr, wd in stages:
        print(f"\n=== Stage: {stage_name} | unfreeze_blocks={unfreeze_blocks} ===")

        model.freeze_encoder()          
        model.encoder_net.train()       
        if args.encoder_type in ["dinov2", "mae", "dinov3"]:
            unfreeze_vit(model, n_blocks=unfreeze_blocks, unfreeze_norm=True)
        else:
            print("WARNING: unfreeze_vit() expects timm ViT (dino/mae). For window_vit you need a different unfreeze helper.")

        optimizer, head_params, enc_params = build_optimizer_param_groups(
            model, head_lr=head_lr, enc_lr=enc_lr, weight_decay=wd
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
        no_improve_epochs = 0

        for _ in range(stage_epochs):
            global_epoch += 1
            print(f"\n--- Epoch {global_epoch}/{args.epochs} (stage {stage_name}) ---")

            model.train()
            running_loss, running_f1 = 0.0, 0.0

            pbar = tqdm(train_loader, desc=f"Training {stage_name} (epoch {global_epoch})")
            for x, masks, edge_masks, meta in pbar:
                x = x.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                edge_masks = edge_masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    total_loss, mask_pred_prob, edge_loss = model(x, masks, edge_masks)

                scaler.scale(total_loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head_params + enc_params, 1.0)

                scaler.step(optimizer)
                scaler.update()

                train_f1 = binary_f1(mask_pred_prob, masks)
                running_loss += total_loss.item()
                running_f1 += train_f1.item()

                pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            scheduler.step()

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_train_f1 = running_f1 / max(len(train_loader), 1)
            wandb.log({
                "stage": stage_name,
                "Train Loss": avg_train_loss,
                "Train F1@0.5": avg_train_f1,
                "lr_head": optimizer.param_groups[0]["lr"],
                "lr_enc": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            }, step=global_epoch)
            print(f"Epoch {global_epoch} - Train loss: {avg_train_loss:.4f} | Train F1@0.5: {avg_train_f1:.4f}")

            if len(val_loader) > 0:
                val_loss, val_f1s, best_t, best_val_f1 = evaluate(model, val_loader, device)

                wandb.log(
                    {f"Val F1@{t}": val_f1s[t] for t in (0.1, 0.3, 0.5)}
                    | {"Val F1 best": best_val_f1, "Best thresh": best_t, "Val loss": val_loss},
                    step=global_epoch
                )
                print(f"Epoch {global_epoch} - Val loss: {val_loss:.4f} | Val F1: {best_val_f1:.4f}")

                # --- early stopping on best F1 ---
                improved = best_val_f1 > (best_f1 + min_delta)
                if improved:
                    best_f1 = best_val_f1
                    no_improve_epochs = 0
                    save_checkpoint(model, optimizer, global_epoch, val_loss, args.save_dir, name + f"_{stage_name}")
                else:
                    no_improve_epochs += 1
                    print(f"[EarlyStop] No improvement for {no_improve_epochs}/{patience} epochs.")

                if no_improve_epochs >= patience:
                    print(f"[EarlyStop] Stopping: Val F1 did not improve for {patience} epochs. Best F1={best_f1:.4f}")
                    wandb.log({"EarlyStop": 1, "EarlyStop_epoch": global_epoch}, step=global_epoch)
                    break

        if no_improve_epochs >= patience:
            break

    wandb.finish()


if __name__ == "__main__":
    main()
