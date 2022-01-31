from collections import OrderedDict

import torch
from segmentation_models_pytorch import UnetPlusPlus

from segmentation.loss_functions import DiceLoss, DiceBCELoss


def get_data_paths():
    return "./ChestX_COVID-main/dataset/masks", \
           "./ChestX_COVID-main/dataset/images", \
           "./ChestX_COVID-main/dataset/labels.csv"


def get_state_dict(net, parameters):
    params_dict = []
    for i, k in enumerate(list(net.state_dict().keys())):
        p = parameters[i]
        if 'num_batches_tracked' in k:
            p = p.reshape(p.size)
        params_dict.append((k, p))
    return OrderedDict({k: torch.Tensor(v) for k, v in params_dict})


def validate(net, val_loader, device):
    criterion = DiceBCELoss()
    # criterion = DiceLoss()
    net.eval()
    val_running_loss = 0.0
    val_running_jac = 0.0
    for batch_idx, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs_masks = net(images)
        loss_seg = criterion(outputs_masks, masks)
        loss = loss_seg

        val_running_loss += loss.item()
        jac = jaccard(outputs_masks.round(), masks)
        val_running_jac += jac.item()

        mask = masks[0, 0, :]
        out = outputs_masks[0, 0, :]
        res = torch.cat((mask, out), 1).cpu().detach()

    val_loss = val_running_loss / len(val_loader)
    val_jac = val_running_jac / len(val_loader)
    return val_loss, val_jac


def get_model():
    return UnetPlusPlus('efficientnet-b4',
                        in_channels=1,
                        classes=1,
                        activation='sigmoid')


def jaccard(outputs, targets):
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)
    union = (outputs + targets).sum(1) - intersection
    jac = (intersection + 0.001) / (union + 0.001)
    return jac.mean()
