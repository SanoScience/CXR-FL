import os
import torch
from collections import OrderedDict
from sklearn.metrics import classification_report, accuracy_score

import torch.nn.functional as F

import torchvision
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")


def accuracy(y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=1)
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def get_state_dict(model, parameters):
    params_dict = []
    for i, k in enumerate(list(model.state_dict().keys())):
        p = parameters[i]
        if 'num_batches_tracked' in k:
            p = p.reshape(p.size)
        params_dict.append((k, p))
    return OrderedDict({k: torch.Tensor(v) for k, v in params_dict})


def get_train_transform_rsna(img_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(img_size, img_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(
            degrees=[-5, 5],
            translate=[0.05, 0.05],
            scale=[0.95, 1.05],
            shear=[-5, 5],
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])


def get_test_transform_rsna(img_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])


def get_model(m, classes=3):
    if m == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=classes)
        model = model.to(DEVICE)
        return model
    if m == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=classes)
        model = model.to(DEVICE)
        return model


def get_data_paths(dataset):
    if 'rsna' in dataset:
        RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
        train_subset = os.path.join(RSNA_DATASET_PATH_BASE, "train_labels_stage_1.csv")
        test_subset = os.path.join(RSNA_DATASET_PATH_BASE, "test_labels_stage_1.csv")
        if dataset == 'rsna-full':
            images_dir = os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images_png/")
        else:
            images_dir = os.path.join(RSNA_DATASET_PATH_BASE, "masked_stage_2_train_images_1024/")
        return images_dir, train_subset, test_subset


def test_single_label(model, device, logger, test_loader, criterion, classes_names):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, batch_label) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            batch_label = batch_label.to(device=device)

            logits = model(image)
            loss = criterion(logits, batch_label)

            test_running_loss += loss.item()
            test_running_accuracy += accuracy(logits, batch_label)

            y_pred = F.softmax(logits, dim=1)
            top_p, top_class = y_pred.topk(1, dim=1)

            test_labels.append(batch_label.view(*top_class.shape))
            test_preds.append(top_class)

            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}\n"
                            f"running_loss: {test_running_loss / (batch_idx + 1):.4f}\n"
                            f"running_acc: {test_running_accuracy / (batch_idx + 1):.4f}\n\n")

    test_preds = torch.cat(test_preds, dim=0).tolist()
    test_labels = torch.cat(test_labels, dim=0).tolist()
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    report_json = json.dumps(
        classification_report(test_labels, test_preds, target_names=classes_names, output_dict=True))

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")

    return test_acc, test_loss, report_json
