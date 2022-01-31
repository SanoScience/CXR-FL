import pandas as pd

import os
from torch.utils.data import Dataset
from PIL import Image

from data_selector import IIDSelector


class RSNADataset(Dataset):
    def __init__(self, client_id, clients_number, ids_labels_file, images_source, transform=None, limit=-1):
        super(RSNADataset, self).__init__()

        self.transform = transform
        # "Normal"=0, "No Lung Opacity / Not Normal"=1, "Lung Opacity"=2
        self.classes_names = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]

        self.ids_labels_df = pd.read_csv(ids_labels_file)

        extension = '.png'

        # IMAGES
        self.images = [os.path.join(images_source, row['patient_id']) + extension for _, row in
                       self.ids_labels_df.iterrows()]
        images_count = len(self.images)
        print(f'Dataset file:{ids_labels_file}, len = {images_count}')

        # LABELS
        self.labels = [row['label'] for _, row in self.ids_labels_df.iterrows()]

        if limit != -1:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

        selector = IIDSelector()
        if client_id != -1:
            self.images, self.labels = selector.select_data(self.images, self.labels, client_id, clients_number)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.labels[idx]
