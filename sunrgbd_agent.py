"""SUN RGB-D loader for the HRFR + RAGS + RGRA training framework.

Returns exactly the fields consumed by train_student.py:
    image           [3,H,W] normalized RGB for DINOv3
    depth           [3,H,W] normalized depth visualization for DINOv3
    geometry_depth  [1,H,W] unnormalized scalar depth for NGFC / NGSplat
    label           [H,W] integer class ID (0 is ignored)
    label_path      str

The three spatial inputs receive identical crop/resize/flip operations.
No arbitrary camera intrinsics are fabricated. If using camera-calibrated 3D
projection, supply correctly transformed per-image intrinsics separately.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from mymodelnew.toolbox.utils import color_map


class SUNRGBD(Dataset):
    def __init__(self, cfg, mode="train", do_aug=True):
        if mode not in ("train", "test"):
            raise ValueError(f"SUNRGBD mode must be train/test; got {mode!r}")
        self.root = Path(cfg["root"])
        self.mode = mode
        self.do_aug = bool(do_aug)
        self.n_classes = int(cfg["n_classes"])
        self.ignore_index = int(cfg.get("id_unlabel", 0))
        self.crop_size = tuple(map(int, cfg["crop_size"].split()))
        if len(self.crop_size) != 2:
            raise ValueError("crop_size must be 'H W'")
        self.scale_range = tuple(map(float, cfg["scales_range"].split()))
        if len(self.scale_range) != 2 or min(self.scale_range) <= 0:
            raise ValueError("scales_range must have two positive values")
        self.flip_prob = float(cfg.get("p", 0.5))
        self.geometry_depth_scale = float(cfg.get("geometry_depth_scale", 1.0))
        # If supplied images are already normalized 8-bit depth, preserve that
        # intensity scale in the DINO input; if 16-bit, use robust normalization.
        self.depth_dino_mode = cfg.get("depth_dino_mode", "auto")
        if self.depth_dino_mode not in ("auto", "robust", "legacy"):
            raise ValueError("depth_dino_mode must be auto / robust / legacy")

        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_normalize = transforms.Normalize(
            [0.449, 0.449, 0.449], [0.226, 0.226, 0.226]
        )
        self.rgb_jitter = transforms.ColorJitter(
            brightness=float(cfg.get("brightness", 0.0)),
            contrast=float(cfg.get("contrast", 0.0)),
            saturation=float(cfg.get("saturation", 0.0)),
        )
        self.class_weight = np.array([
            4.1489, 4.7033, 6.0432, 21.9638, 19.1237, 11.3568,
            24.0552, 15.1227, 23.6196, 22.7505, 39.8595, 38.5019,
            36.881, 42.1598, 27.4841, 42.7398, 34.586, 40.8405,
            39.3031, 39.7861, 49.9436, 43.8466, 34.9577, 43.7357,
            45.1824, 46.2051, 43.9607, 46.3582, 49.9228, 40.8575,
            39.0468, 48.2333, 48.3639, 44.6668, 42.9215, 45.1362,
            45.8617, 45.8814,
        ], dtype=np.float32)
        if self.class_weight.size != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} class weights; got {self.class_weight.size}"
            )

        image_dir = self.root / mode / "image"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"SUN RGB-D path not found: {image_dir}. "
                "Check 'root' in sunrgbd.json and train/test/image layout."
            )
        # Discover actual indices; old code hardcoded 5285/5050 samples and could
        # try loading missing files if the local copy has another split size.
        paths = sorted(
            (p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
            key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem),
        )
        if not paths:
            raise RuntimeError(f"No RGB images in {image_dir}")
        self.ids = [p.stem for p in paths]
        self.image_paths = paths
        depth_dir = self.root / mode / "depth"
        label_dir = self.root / mode / "label"
        missing = [
            stem for stem in self.ids
            if not (depth_dir / f"{stem}.png").is_file()
            or not (label_dir / f"{stem}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} SUN RGB-D samples are missing matching "
                f"depth/label PNGs; examples: {missing[:5]}"
            )

    def __len__(self):
        return len(self.ids)

    def _augment(self, rgb, depth, label):
        """Joint geometric transforms; use nearest neighbor on depth and label."""
        # Replicate the original loader's Resize(crop_size) first.
        target_h, target_w = self.crop_size
        rgb = TF.resize(rgb, [target_h, target_w], interpolation=InterpolationMode.BILINEAR)
        depth = TF.resize(depth, [target_h, target_w], interpolation=InterpolationMode.NEAREST)
        label = TF.resize(label, [target_h, target_w], interpolation=InterpolationMode.NEAREST)

        if self.mode == "train" and self.do_aug:
            if torch.rand(()) < self.flip_prob:
                rgb, depth, label = (TF.hflip(x) for x in (rgb, depth, label))
            min_scale, max_scale = self.scale_range
            scale = float(torch.empty(()).uniform_(min_scale, max_scale))
            scaled_h = max(1, int(round(target_h * scale)))
            scaled_w = max(1, int(round(target_w * scale)))
            if (scaled_h, scaled_w) != (target_h, target_w):
                rgb = TF.resize(rgb, [scaled_h, scaled_w], interpolation=InterpolationMode.BILINEAR)
                depth = TF.resize(depth, [scaled_h, scaled_w], interpolation=InterpolationMode.NEAREST)
                label = TF.resize(label, [scaled_h, scaled_w], interpolation=InterpolationMode.NEAREST)
            # Pad all modalities with aligned offsets; 0 means invalid geometry.
            pad_bottom = max(0, target_h - scaled_h)
            pad_right = max(0, target_w - scaled_w)
            if pad_bottom or pad_right:
                pad = [0, 0, pad_right, pad_bottom]
                rgb = TF.pad(rgb, pad, fill=0)
                depth = TF.pad(depth, pad, fill=0)
                label = TF.pad(label, pad, fill=self.ignore_index)
            w, h = rgb.size
            top = int(torch.randint(h - target_h + 1, (1,)).item())
            left = int(torch.randint(w - target_w + 1, (1,)).item())
            rgb, depth, label = (
                TF.crop(x, top, left, target_h, target_w)
                for x in (rgb, depth, label)
            )
            # Photometric jitter applies ONLY to RGB, not geometry or labels.
            rgb = self.rgb_jitter(rgb)
        return rgb, depth, label

    @staticmethod
    def _scalar_depth_array(depth_image):
        raw = np.asarray(depth_image)
        if raw.ndim == 3:
            if raw.shape[2] < 3 or not np.array_equal(raw[..., 0], raw[..., 1]) or not np.array_equal(raw[..., 0], raw[..., 2]):
                raise ValueError(
                    "Depth PNG appears color-coded; an RGB image cannot recover "
                    "true scalar geometry. Supply a one-channel depth PNG."
                )
            raw = raw[..., 0]
        if raw.ndim != 2:
            raise ValueError(f"Depth image must be scalar, got shape {raw.shape}")
        return np.asarray(raw, dtype=np.float32)

    def _to_depth_tensors(self, depth_image):
        raw = self._scalar_depth_array(depth_image)
        # Preserve original depth values for NGFC/3DGS (no DINO Normalize).
        geometry = raw * self.geometry_depth_scale
        geometry = np.where(np.isfinite(geometry) & (geometry > 0), geometry, 0.0)
        geometry_depth = torch.from_numpy(geometry.copy()).unsqueeze(0)

        valid = raw[np.isfinite(raw) & (raw > 0)]
        if valid.size == 0:
            normalized = np.zeros_like(raw, dtype=np.float32)
        elif self.depth_dino_mode == "legacy" or (
            self.depth_dino_mode == "auto" and valid.max() <= 255
        ):
            normalized = np.clip(raw / 255.0, 0, 1)
        else:
            # 16-bit scalar depth should not be converted directly to 8-bit RGB.
            # Robust normalization affects only the DINO feature stream.
            high = max(float(np.percentile(valid, 99)), 1.0)
            normalized = np.clip(raw / high, 0, 1)
        normalized[~(np.isfinite(raw) & (raw > 0))] = 0.0
        dino_depth = torch.from_numpy(normalized.copy()).float().unsqueeze(0).repeat(3, 1, 1)
        dino_depth = self.dp_normalize(dino_depth)
        return dino_depth, geometry_depth

    def __getitem__(self, index):
        stem = self.ids[index]
        rgb_path = self.image_paths[index]
        depth_path = self.root / self.mode / "depth" / f"{stem}.png"
        label_path = self.root / self.mode / "label" / f"{stem}.png"
        with Image.open(rgb_path) as im:
            rgb = im.convert("RGB")
        with Image.open(depth_path) as im:
            depth = im.copy()  # IMPORTANT: no .convert('RGB') before geometry
        with Image.open(label_path) as im:
            label = im.copy()  # preserve class-index palette or grayscale mode
        rgb, depth, label = self._augment(rgb, depth, label)

        depth_dino, geometry_depth = self._to_depth_tensors(depth)
        label_arr = np.asarray(label, dtype=np.int64)
        if label_arr.ndim != 2:
            raise ValueError(f"Expected single-channel index labels, got {label_arr.shape}")
        if (label_arr < 0).any() or (label_arr >= self.n_classes).any():
            raise ValueError(
                f"Label out of range [0,{self.n_classes-1}] in {label_path}; "
                f"range [{label_arr.min()}, {label_arr.max()}]"
            )
        return {
            "image": self.im_to_tensor(rgb),
            "depth": depth_dino,
            "geometry_depth": geometry_depth,
            "label": torch.from_numpy(label_arr.copy()).long(),
            "label_path": label_path.name,
        }

    @property
    def cmap(self):
        return color_map(N=self.n_classes)
