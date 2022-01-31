from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, densenet121
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import PIL
import time

st = time.time()
device = torch.device('cpu')


def get_resnet_cam(model_path):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device))
    target_layer_resnet = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer_resnet], use_cuda=False)
    return cam, model


def get_densenet_cam(model_path):
    model = densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(in_features=1024, out_features=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device))
    target_layer_densenet = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer_densenet], use_cuda=False)
    return cam, model


def get_image(path):
    return PIL.Image.open(path).convert('RGB')


def convert_img(image_rgb):
    convert = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        # Normalize to ImageNet
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    input_tensor = convert(image_rgb).float()

    v = Variable(input_tensor, requires_grad=True)
    v = v.unsqueeze(0)
    return v


def get_visualization(path_to_image, cam_obj):
    image_rgb = get_image(path_to_image)
    converted_img = convert_img(image_rgb)
    grayscale_cam = cam_obj(input_tensor=converted_img)
    res_conv = torchvision.transforms.Resize(size=(224, 224))
    image_rgb = res_conv(image_rgb)
    img_np = np.array(image_rgb) / 255
    visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
    return visualization


images = [('./FederatedLearning_MSc/segmented.png',
           './FederatedLearning_MSc/full.png')]

for image_path_tuple in images:
    image_path_segmented, image_path_full = image_path_tuple
    cam_resnet_segmented, resnet_segmented = get_resnet_cam(
        './ResNet50_segmented')
    cam_resnet_full, resnet_full = get_resnet_cam(
        './ResNet50_full')
    cam_densenet_segmented, densenet_segmented = get_densenet_cam(
        './DenseNet121_segmented')
    cam_densenet_full, densenet_full = get_densenet_cam(
        './DenseNet121_full')

    converted_img_segmented = convert_img(get_image(image_path_segmented))
    converted_img_full = convert_img(get_image(image_path_full))

    pred1 = np.argmax(resnet_segmented(converted_img_segmented).detach().numpy())
    pred2 = np.argmax(densenet_segmented(converted_img_segmented).detach().numpy())
    pred3 = np.argmax(resnet_full(converted_img_full).detach().numpy())
    pred4 = np.argmax(densenet_full(converted_img_full).detach().numpy())

    print(pred1, pred2, pred3, pred4)
    pred_set = set([pred1, pred2, pred3, pred4])
    if len(pred_set) != 1:
        continue
    classes = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]
    print("Predicted class: ", classes[pred1])
    vis1 = get_visualization(image_path_segmented, cam_resnet_segmented)
    vis2 = get_visualization(image_path_full, cam_resnet_full)
    vis3 = get_visualization(image_path_segmented, cam_densenet_segmented)
    vis4 = get_visualization(image_path_full, cam_densenet_full)
    plt.figure()
    plt.axis('off')
    f, axarr = plt.subplots(1, 4)

    fig1 = axarr[0].imshow(vis1)
    fig2 = axarr[1].imshow(vis2)
    fig3 = axarr[2].imshow(vis3)
    fig4 = axarr[3].imshow(vis4)

    figs = [fig1, fig2, fig3, fig4]
    for f in figs:
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

et = time.time()

print(et - st)
