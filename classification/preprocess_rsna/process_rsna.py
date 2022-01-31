import pandas as pd

import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import pydicom
import numpy as np

import torch
import torchvision
from segmentation_models_pytorch import UnetPlusPlus

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ImageFile.LOAD_TRUNCATED_IMAGES = True

RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
SEGMENTATION_MODEL = "best_model/unet"
IMAGES_DIR = os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images/")
LABELS = os.path.join(RSNA_DATASET_PATH_BASE, "train_labels_stage_1.csv")


class RSNADataset(Dataset):
    def __init__(self, ids_labels_file, images_source):
        super(RSNADataset, self).__init__()

        # "Normal"=0, "No Lung Opacity / Not Normal"=1, "Lung Opacity"=2
        self.classes_names = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]

        self.ids_labels_df = pd.read_csv(ids_labels_file)

        extension = '.dcm'

        # IMAGES
        self.images = [os.path.join(images_source, row['patient_id']) + extension for _, row in
                       self.ids_labels_df.iterrows()]
        images_count = len(self.images)
        print(f'Dataset file:{ids_labels_file}, len = {images_count}')

        # LABELS
        self.labels = [row['label'] for _, row in self.ids_labels_df.iterrows()]

    def __len__(self):
        return len(self.labels)

    def get_image(self, img_path):
        """Load a dicom image to an array"""
        try:
            dcm_data = pydicom.read_file(img_path)
            img = dcm_data.pixel_array
            return img
        except:
            pass

    def __getitem__(self, idx):
        image_path = self.images[idx]
        im_array = self.get_image(image_path)
        image_rgb = Image.fromarray(im_array).convert('RGB')
        patient_id = self.ids_labels_df.iloc[idx]['patient_id']
        image_rgb.save(
            f'./RSNA/stage_2_train_images_png/{patient_id}.png',
            'PNG')
        image_l = image_rgb.convert("L")
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(1024, 1024)),
            torchvision.transforms.ToTensor(),
        ])
        return trans(image_l), self.labels[idx]


segmentation_model = UnetPlusPlus('efficientnet-b4', in_channels=1, classes=1, activation='sigmoid').to(DEVICE)
segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL, map_location=torch.device('cpu')))

rsna_dataset = RSNADataset(LABELS, IMAGES_DIR)

test_patching_loader = torch.utils.data.DataLoader(rsna_dataset, batch_size=1, num_workers=8, pin_memory=True)

segmentation_model.eval()
for image_idx, (image, batch_label) in enumerate(test_patching_loader):
    image = image.to(DEVICE)
    with torch.no_grad():
        outputs_mask = segmentation_model(image)

    image = image[0, 0, :]
    img_np = image.cpu().numpy()
    out = outputs_mask[0, 0, :]
    out_np = out.cpu().numpy()

    superposed = np.copy(img_np)
    superposed[out_np < 0.05] = 0

    patient_id = rsna_dataset.ids_labels_df.iloc[image_idx]['patient_id']

    Image.fromarray((255 * superposed).astype(np.int8), mode='L').convert('RGB').save(
        os.path.join(RSNA_DATASET_PATH_BASE, f"masked_stage_2_train_images_{1024}/",
                     f"{patient_id}.png"), 'PNG')

    if image_idx % 50 == 0:
        print(f"batch_idx: {image_idx}")
