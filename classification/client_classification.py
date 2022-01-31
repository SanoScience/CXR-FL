import logging
import time

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score

from fl_rsna_dataset import RSNADataset

from utils import get_state_dict, accuracy, get_train_transform_rsna, get_model, get_data_paths

import torch.nn.functional as F
import click

IMAGE_SIZE = 224
LIMIT = -1

hdlr = logging.StreamHandler()
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(hdlr)
LOGGER.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_single_label(model, train_loader, criterion, optimizer, classes_names, epochs):
    for epoch in range(epochs):
        start_time_epoch = time.time()
        LOGGER.info(f"Starting epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        preds = []
        labels = []

        for batch_idx, (images, batch_labels) in enumerate(train_loader):
            images = images.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(logits, batch_labels)

            y_pred = F.softmax(logits, dim=1)
            top_p, top_class = y_pred.topk(1, dim=1)

            labels.append(batch_labels.view(*top_class.shape))
            preds.append(top_class)

            if batch_idx % 10 == 0:
                LOGGER.info(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                            f" Loss: {running_loss / (batch_idx + 1):.4f}"
                            f" Acc: {running_accuracy / (batch_idx + 1):.4f}"
                            f" Time: {time.time() - start_time_epoch:2f}")
        preds = torch.cat(preds, dim=0).tolist()
        labels = torch.cat(labels, dim=0).tolist()
        LOGGER.info("Training report:")
        LOGGER.info(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(labels, preds)

        LOGGER.info(f" Training Loss: {train_loss:.4f}"
                    f" Training Acc: {train_acc:.4f}")


def load_data(client_id, clients_number, d_name, bs):
    if 'rsna' in d_name:
        images_dir, train_subset, _ = get_data_paths(d_name)
        LOGGER.info(f"images_dir: {images_dir}")
        train_transform = get_train_transform_rsna(IMAGE_SIZE)
        train_dataset = RSNADataset(client_id, clients_number, train_subset, images_dir, transform=train_transform,
                                    limit=LIMIT)
        return torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=8,
                                           pin_memory=True), train_dataset.classes_names


class SingleLabelClassificationClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number, m_name):
        # Load model
        self.model = get_model(m_name)
        self.client_id = client_id
        self.clients_number = clients_number
        self.train_loader = None
        self.classes_names = None

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        LOGGER.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        LOGGER.info("Parameters loaded")

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        batch_size = int(config["batch_size"])
        epochs = int(config["local_epochs"])
        lr = float(config["learning_rate"])
        d_name = config["dataset_type"]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.00001)

        if not self.train_loader:
            self.train_loader, self.classes_names = load_data(self.client_id, self.clients_number, d_name, batch_size)

        train_single_label(self.model, self.train_loader, criterion, optimizer, self.classes_names, epochs=epochs)
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        pass


@click.command()
@click.option('--sa', default='', type=str, help='Server address')
@click.option('--c_id', default=0, type=int, help='Client id')
@click.option('--c', default=1, type=int, help='Clients number')
@click.option('--m', default='ResNet50', type=str, help='Model used for training')
def run_client(sa, c_id, c, m):
    # Start client
    LOGGER.info("Connecting to:" + f"{sa}:8087")
    fl.client.start_numpy_client(f"{sa}:8087",
                                 client=SingleLabelClassificationClient(c_id, c, m))


if __name__ == "__main__":
    run_client()
