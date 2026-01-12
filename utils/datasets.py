import os
from typing import List, Tuple, Optional, Dict, Sequence, Union

import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm  

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F
from torchvision.io import read_image

class InpaintingSegDataset(Dataset):
    """
    Dataset for your inpainting segmentation setup.

    Supports mixing multiple inpainting methods (STTN/OPN) in one dataset.
    For each sample, it yields:
        x:          [3, H, W]   (inpainted frame for a specific method)
        mask:       [1, H, W]   (binary GT mask)
        edge_mask:  [1, H, W]   (currently all ones)
        meta:       dict with video_id, frame_name, method
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        split: str,
        methods: Union[str, Sequence[str]] = "STTN",
        input_size: int = 240,
        frame_res_dir: str = "432x240",
        mask_res_dir: str = "resized_432x240",
        img_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Args:
            csv_path: path to dataset.csv
            root_dir: root folder containing STTN/, OPN/, input_frames/, input_masks/
            split: 'train', 'val', or 'test'
            methods: one of:
                     - "STTN" / "OPN"
                     - list/tuple of these, e.g. ["STTN", "OPN"]
                     - "all" -> ["STTN", "OPN"]
            input_size: final square size for the model (e.g. 240 or 1024)
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.input_size = input_size
        self.frame_res_dir = frame_res_dir
        self.mask_res_dir = mask_res_dir
        self.img_extensions = img_extensions
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        # --- normalize methods argument ---
        if isinstance(methods, str):
            if methods.lower() == "all":
                methods_list = ["STTN", "OPN"]
            else:
                methods_list = [methods]
        else:
            methods_list = list(methods)

        # keep canonical capitalization
        self.methods: List[str] = [m.upper() for m in methods_list]

        # --- Load and filter CSV ---
        df = pd.read_csv(csv_path)

        # keep only rows that correspond to real samples
        split_col = df["Unnamed: 1"].astype(str).str.lower()    # Select split column
        df = df[split_col.isin(["train", "val", "test"])]   # Keep only valid splits
        df_split = df[split_col == self.split]  # Keep only rows for the desired split



        if df_split.empty:
            raise RuntimeError(
                f"No IDs found for split='{split}' with methods={self.methods}. "
                f"Check CSV content and method names."
            )

        # --- Build list of frame-level samples (with progress bar) ---
        self.samples: List[Dict] = []

        count = 0

        desc = f"Indexing {self.split} (methods={','.join(self.methods)})"
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=desc):
            vid = str(row["ID"])

            for m in self.methods:
             
                inpaint_dir = os.path.join(root_dir, m, frame_res_dir, vid) # inpainted frames by method m
                orig_dir = os.path.join(root_dir, "input_frames", frame_res_dir, vid) # original frames
                mask_dir = os.path.join(root_dir, "input_masks", self.mask_res_dir, vid) # GT masks

                if not os.path.isdir(inpaint_dir):
                    # optionally warn / skip if folder does not exist
                    # print(f"[WARN] inpaint dir not found: {inpaint_dir}")
                    continue

                for fname in sorted(os.listdir(inpaint_dir)):
                    # skip non-image files
                    if not fname.lower().endswith(self.img_extensions):
                        continue

                    inp_path = os.path.join(inpaint_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)
                    orig_path = os.path.join(orig_dir, fname)
                    
                    #skip if inpainted file does not exist
                    if not os.path.isfile(inp_path):
                        print(f"[WARN] missing inpainted frame for {vid}/{fname} ({m})")
                        continue

                    # skip if mask file does not exist
                    if not os.path.isfile(mask_path):
                        print(f"[WARN] missing mask for {vid}/{fname} ({m})")
                        continue

                    # add sample to list
                    self.samples.append(
                        dict(
                            inpaint_path=inp_path,
                            mask_path=mask_path,
                            orig_path=orig_path,
                            video_id=vid,
                            frame_name=fname,
                            method=m,
                        )
                    )
                
                    count += 1
                    
                    
        
        print(f"Total frames indexed for split='{split}' and methods={self.methods}: {count}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No frame samples found on disk for split='{split}' and methods={self.methods}. "
                f"Check folder structure & file names."
            )

        # --- Define transforms ---
        # TODO: add padding here
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)), # resize to input size
                # Since ToTensor() will be deprecated next release
                transforms.ToImage(), # convert to image
                transforms.ToDtype(torch.float32, scale=True), # scale to [0,1]
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)), # resize to input size
                # Since ToTensor() will be deprecated next release
                transforms.ToImage(), # convert to image
                transforms.ToDtype(torch.float32, scale=True), # scale to [0,1]      
            ]
        )
        self.padding_transform = albu.Compose([
            albu.PadIfNeeded(          
                min_height=input_size,
                min_width=input_size, 
                border_mode=0, 
                # value=0, 
                position= 'top_left',
                # mask_value=0
                ),
            # albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, input_size, input_size), # If pad is not needed, crop to input size
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
            rec = self.samples[idx]

            # uint8 tensors, CHW
            x = read_image(rec["inpaint_path"])          # [3,H,W], uint8
            m = read_image(rec["mask_path"])             # [1,H,W] or [3,H,W], uint8

            if m.shape[0] != 1:
                m = m[:1]  # keep 1 channel

            H, W = x.shape[1], x.shape[2]
            S = self.input_size

            # pad bottom/right to at least S (top-left anchored)
            pad_h = max(0, S - H)
            pad_w = max(0, S - W)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
                m = F.pad(m, (0, pad_w, 0, pad_h), value=0)

            # crop top-left to SxS 
            x = x[:, :S, :S]
            m = m[:, :S, :S]

            # convert/scaling
            x = x.float().div_(255.0)                
            x = (x - self.imagenet_mean) / self.imagenet_std
            mask = (m > 0).float()              
           

            edge_mask = torch.ones_like(mask)

            meta = {
                "video_id": rec["video_id"],
                "frame_name": rec["frame_name"],
                "method": rec["method"],
            }
            return x, mask, edge_mask, meta


def make_dataloader(
    csv_path: str,
    root_dir: str,
    split: str,
    methods: Union[str, Sequence[str]] = "STTN",
    input_size: int = 240,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> DataLoader:

    if shuffle is None:
        shuffle = split.lower() == "train"

    dataset = InpaintingSegDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        split=split,
        methods=methods,
        input_size=input_size,
    )

    # print x.shape, x.dtype, mask.shape, mask.dtype
    print(f"Sample from {split} dataset: x.shape={dataset[0][0].shape}, x.dtype={dataset[0][0].dtype}, mask.shape={dataset[0][1].shape}, mask.dtype={dataset[0][1].dtype}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return loader
def build_test_dataloader(
    csv_path: str,
    root_dir: str,
    methods: Union[str, Sequence[str]] = "STTN",
    input_size: int = 240,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    return make_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        split="test",
        methods=methods,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

def build_all_dataloaders(
    csv_path: str,
    root_dir: str,
    methods: Union[str, Sequence[str]] = "STTN",
    input_size: int = 240,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_loader = make_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        split="train",
        methods=methods,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = make_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        split="val",
        methods=methods,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = make_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        split="test",
        methods=methods,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Total samples loaded:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val:   {len(val_loader.dataset)}")
    print(f"  Test:  {len(test_loader.dataset)}")
    print(f"  TOTAL: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    print(f"Total batches created:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val:   {len(val_loader)}")
    print(f"  Test:  {len(test_loader)}")

    # print x.shape, x.dtype, mask.shape, mask.dtype
    print(f"Sample from train dataset: x.shape={train_loader.dataset[0][0].shape}, x.dtype={train_loader.dataset[0][0].dtype}, mask.shape={train_loader.dataset[0][1].shape}, mask.dtype={train_loader.dataset[0][1].dtype}")
    print(f"Sample from val dataset: x.shape={val_loader.dataset[0][0].shape}, x.dtype={val_loader.dataset[0][0].dtype}, mask.shape={val_loader.dataset[0][1].shape}, mask.dtype={val_loader.dataset[0][1].dtype}")
    print(f"Sample from test dataset: x.shape={test_loader.dataset[0][0].shape}, x.dtype={test_loader.dataset[0][0].dtype}, mask.shape={test_loader.dataset[0][1].shape}, mask.dtype={test_loader.dataset[0][1].dtype}")

    return train_loader, val_loader, test_loader


"""
Code below is for testing
"""

if __name__ == "__main__":
    print("Testing InpaintingSegDataset...")
   
