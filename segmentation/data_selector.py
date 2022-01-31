import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import os

logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

class DataSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_client_data(self, images, masks, client_id, number_of_clients, labels_dict):
        pass

    @abstractmethod
    def select_server_data(self, images, masks, labels_dict):
        pass


class IIDSelector(DataSelector):
    def prepare_order(self, images, masks, labels_df):
        data = defaultdict(list)
        image_to_mask = self.get_image_to_mask_map(images, masks)
        res_images = []
        res_masks = []
        res_labels = []
        for index, row in labels_df.iterrows():
            if row['train']:
                data[row['class']].append(row['filename'])

        for cls, imgs in data.items():
            for img in imgs:
                if img in image_to_mask and cls != 'no_class':
                    res_img, res_mask = image_to_mask[img]
                    res_images.append(res_img)
                    res_masks.append(res_mask)
                    res_labels.append(cls)

        return res_images, res_masks, res_labels

    def get_image_to_mask_map(self, images, masks):
        image_to_mask = {}
        for i, m in zip(images, masks):
            image_to_mask[os.path.basename(i)] = (i, m)
        return image_to_mask

    def select_client_data(self, images, masks, client_id, number_of_clients, labels_df):
        ordered_images, ordered_masks, labels = self.prepare_order(images, masks, labels_df)
        sampled_images = [path for i, path in enumerate(ordered_images) if (i % number_of_clients) == client_id]
        sampled_masks = [path for i, path in enumerate(ordered_masks) if (i % number_of_clients) == client_id]
        sampled_labels = [path for i, path in enumerate(labels) if (i % number_of_clients) == client_id]
        counter = Counter()
        for l in sampled_labels:
            counter[l] += 1

        zipped = list(zip(sampled_images, sampled_masks))
        random.shuffle(zipped)
        sampled_images, sampled_masks = zip(*zipped)
        logger.info(f"Size of data: {len(sampled_images)}")
        logger.info(f"Distribution of data: {str(counter)}")
        return sampled_images, sampled_masks

    def select_server_data(self, images, masks, labels_df):
        test_imgs, test_masks, labels = [], [], []
        image_to_mask = self.get_image_to_mask_map(images, masks)
        for index, row in labels_df.iterrows():
            filename = row['filename']
            label = row['class']
            is_train = row['train']
            basename = os.path.basename(filename)
            if basename in image_to_mask and not is_train and label != "no_class":
                i, m = image_to_mask[basename]
                test_imgs.append(i)
                test_masks.append(m)
                labels.append(label)
        counter = Counter()
        for l in labels:
            counter[l] += 1
        logger.info(f"Size of data: {len(test_imgs)}")
        logger.info(f"Distribution of data: {str(counter)}")

        return test_imgs, test_masks
