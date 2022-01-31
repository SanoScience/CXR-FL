from abc import ABC, abstractmethod


class DataSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_data(self, images, masks, client_id, number_of_clients):
        pass


class IIDSelector(DataSelector):
    def select_data(self, images, labels, client_id, number_of_clients):
        sampled_images = [path for i, path in enumerate(images) if i % number_of_clients == client_id]
        sampled_labels = [label for i, label in enumerate(labels) if i % number_of_clients == client_id]
        return sampled_images, sampled_labels
