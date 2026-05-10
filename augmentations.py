import os 
import sys 
import glob 
import re 
import cv2 
import json

import torch 
import PIL
import numpy as np 
import matplotlib.pyplot as plt 
import albumentations as A 
import torchvision.datasets as tds
import torchvision.transforms as transforms 
import skimage as sk 
from tqdm import tqdm

from PIL import Image
from scipy.ndimage import zoom as scizoom 
from skimage.filters import gaussian
from io import BytesIO

class BasicAugmentation:
    def __init__(self, input_dir, output_dir, size=(512, 512)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.size = size
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.config = {
            "rotate": [15, 30, 45],
            "severities": [1, 2, 3],
            "crop_resize": [0.8, 1.0],
            "shear": [15],
            "color_jitter": {
                "brightness": [0.2],
                "contrast": [0.2],
                "saturation": [0.2],
                "hue": [0.1],
            },
        }

        self.transforms = {
            "gaussian": self._gaussian_noise,
            "jpeg": self._jpeg_noise,
            "color_jitter": self._color_jitter,
            "color_space": self._color_space,
            "random_rotation": self._random_rotation,
            "flip": self._flip,
            "translate": self._translate,
            "shear": self._shear,
            "cropping": self._cropping,
        }
        self.metadata = []

    def _to_numpy(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image

    def _read_image_path(self, img_path):
        image = Image.open(img_path)
        image = image.resize(size=self.size)
        return image

    def _save_image(self, image, file_name, subdir=None):
        if subdir is not None:
            save_dir = os.path.join(self.output_dir, subdir)
        else:
            save_dir = self.output_dir

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        image.save(save_path)
        return save_path

    def _apply_transform(self, image, transform_name, transform_func):
        if transform_name in ("gaussian", "jpeg"):
            severity = int(self.rng.choice(self.config["severities"]))
            return transform_func(image, severity=severity)
        return transform_func(image)

    def _write_metadata_json(self, metadata, file_name="metadata.json"):
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return out_path

    def _gaussian_noise(self, image, severity: int = 1):
        temp_image = image

        if severity == 1:
            std_range = (0.05, 0.10)
        elif severity == 2:
            std_range = (0.10, 0.18)
        elif severity == 3:
            std_range = (0.18, 0.30)
        else:
            raise ValueError("severity must be in [1, 3]")

        transform = A.GaussNoise(std_range=std_range, p=1)
        transformed_image = transform(image=temp_image)["image"]

        return transformed_image

    def _jpeg_noise(self, image, severity: int = 1):
        c = [25, 18, 12][severity - 1]

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image = image.astype(np.uint8)

        buffer = BytesIO()

        Image.fromarray(image).save(buffer, format="JPEG", quality=c)

        transformed_image = np.array(Image.open(buffer))
        return transformed_image

    def _color_jitter(self, image):
        image = self._to_numpy(image)
        cfg = self.config["color_jitter"]
        transform = A.ColorJitter(
            brightness=cfg["brightness"][0],
            contrast=cfg["contrast"][0],
            saturation=cfg["saturation"][0],
            hue=cfg["hue"][0],
            p=1,
        )
        return transform(image=image)["image"]

    def _random_rotation(self, image):
        image = self._to_numpy(image)
        angle = int(self.rng.choice(self.config["rotate"]))
        transform = A.Rotate(limit=(angle, angle), p=1)
        return transform(image=image)["image"]

    def _flip(self, image):
        image = self._to_numpy(image)
        if self.rng.random() < 0.5:
            transform = A.HorizontalFlip(p=1)
        else:
            transform = A.VerticalFlip(p=1)
        return transform(image=image)["image"]

    def _translate(self, image, max_shift=0.1):
        image = self._to_numpy(image)
        shift_x = float(self.rng.uniform(-max_shift, max_shift))
        shift_y = float(self.rng.uniform(-max_shift, max_shift))
        transform = A.Affine(translate_percent={"x": shift_x, "y": shift_y}, p=1)
        return transform(image=image)["image"]

    def _shear(self, image):
        image = self._to_numpy(image)
        angle = int(self.rng.choice(self.config["shear"]))
        transform = A.Affine(shear={"x": angle, "y": angle}, p=1)
        return transform(image=image)["image"]

    def _cropping(self, image):
        image = self._to_numpy(image)
        scale = self.config["crop_resize"]
        transform = A.RandomResizedCrop(
            size=(self.size[0], self.size[1]),
            scale=(scale[0], scale[1]),
            ratio=(0.9, 1.1),
            p=1,
        )
        return transform(image=image)["image"]

    def _color_space(self, image, mode=None):
        image = self._to_numpy(image)
        if mode is None:
            mode = str(self.rng.choice(["hsv", "lab", "gray"]))

        if mode == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if mode == "lab":
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if mode == "gray":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return np.stack([gray, gray, gray], axis=-1)

        raise ValueError("mode must be in ['hsv', 'lab', 'gray']")

    def _random_two_augmentations(self, image):
        keys = list(self.transforms.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two transforms to combine")

        first, second = self.rng.choice(keys, size=2, replace=False)
        img_first = self._apply_transform(image, first, self.transforms[first])
        img_first = self._to_numpy(img_first)

        img_second = self._apply_transform(img_first, second, self.transforms[second])
        return img_second, (first, second)

    def apply_augmentation(self, include_random_two=False, random_two_count=1):
        img_names = os.listdir(self.input_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for img_name in tqdm(img_names):
            img_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.input_dir, img_name)

            if not os.path.isfile(img_path):
                continue
            if os.path.splitext(img_name)[1].lower() not in valid_ext:
                continue

            original_image = self._read_image_path(img_path=img_path)
            base_image = self._to_numpy(original_image)

            for transform, transform_func in self.transforms.items():
                aug = self._apply_transform(base_image, transform, transform_func)

                if isinstance(aug, np.ndarray):
                    if aug.dtype != np.uint8:
                        aug = np.clip(aug, 0, 255).astype(np.uint8)
                    aug_img = Image.fromarray(aug)
                else:
                    aug_img = aug

                file_name = f"{img_id}_{transform}.png"
                save_path = self._save_image(aug_img, file_name, subdir=transform)
                self.metadata.append(
                    {
                        "img_id": img_id,
                        "transform": transform,
                        "save_path": save_path,
                    }
                )

            if include_random_two:
                for idx in range(int(random_two_count)):
                    aug, (t1, t2) = self._random_two_augmentations(base_image)
                    if isinstance(aug, np.ndarray):
                        if aug.dtype != np.uint8:
                            aug = np.clip(aug, 0, 255).astype(np.uint8)
                        aug_img = Image.fromarray(aug)
                    else:
                        aug_img = aug

                    combo_name = f"random2_{t1}_{t2}"
                    file_name = f"{img_id}_{combo_name}_{idx + 1}.png"
                    save_path = self._save_image(aug_img, file_name, subdir="random_two")
                    self.metadata.append(
                        {
                            "img_id": img_id,
                            "transform": combo_name,
                            "save_path": save_path,
                        }
                    )

        self._write_metadata_json(self.metadata)
        return self.metadata
