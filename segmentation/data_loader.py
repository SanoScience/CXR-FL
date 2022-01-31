"""Data loader for lung segmentation."""
from segmentation.data_selector import IIDSelector

import os
import glob
from PIL import Image
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset
import pandas as pd


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    return int(int_part)


class LungSegDataset(Dataset):
    """class LungSegDataset."""

    def __init__(self,
                 client_id: int = None,
                 clients_number: int = None,
                 path_to_images: str = None,
                 path_to_masks: str = None,
                 image_size: int = None,
                 mode: str = None,
                 labels: str = None) -> None:
        """
        Args:
            path_to_images:
            path_to_masks:
            image_size:
        """
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.image_size = image_size
        self.mode = mode
        selector = IIDSelector()
        imgs = sorted(glob.glob(self.path_to_images + "/*.jpeg"), key=get_key)
        masks = sorted(glob.glob(self.path_to_masks + "/*.png"), key=get_key)
        labels_df = pd.read_csv(labels)
        if mode == 'test':
            self.images, self.masks = selector.select_server_data(imgs, masks, labels_df)
        else:
            self.images, self.masks = selector.select_client_data(imgs, masks, client_id, clients_number, labels_df)

    def __getitem__(self, x) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x:
        Returns:
        """
        image = Image.open(self.images[x]).convert("L")
        mask = Image.open(self.masks[x])
        resize_transform = transforms.Resize(size=(self.image_size, self.image_size))
        image = resize_transform(image)
        mask = resize_transform(mask)

        if self.mode == "train":
            if random.random() > 0.5:
                color_jitter_transform = transforms.ColorJitter(
                    brightness=[0.8, 1.2],
                    contrast=[0.8, 1.2],
                    saturation=[0.8, 1.2],
                    hue=[-0.1, 0.1]
                )
                image = color_jitter_transform.forward(image)

            if random.random() > 0.5:
                (angle, translations, scale, shear) = transforms.RandomAffine.get_params(
                    degrees=[-90, 90],
                    translate=[0.2, 0.2],
                    scale_ranges=[1, 2],
                    shears=[-10, 10],
                    img_size=[self.image_size, self.image_size]
                )
                image = F.affine(
                    img=image,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    fill=0
                )
                mask = F.affine(
                    mask,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    fill=0
                )

            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.images)
