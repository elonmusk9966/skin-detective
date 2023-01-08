import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gdown
import os
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
import base64
import numpy as np
from PIL import Image
from io import BytesIO


img_size = (224, 224)

# Label information
label2id = {'ez': 0, 'ps': 1, 'others': 2, 'normal': 3}
id2label = {v: k for k, v in label2id.items()}


def download_model(url, output='model.pt'):
  gdown.download(url, output, quiet=False)


if not os.path.exists('./model.pt'):
  # Model v1
  #  download_model("https://drive.google.com/u/0/uc?id=1GH-YFFloEULAEU3bk55iLJkRV3z9hbKL&export=download&confirm=t")
  # Model v2
  download_model(
      "https://drive.google.com/u/0/uc?id=1ZpWItLNAxlMj0nNGcPldGpl6meV6rSbO&export=download&confirm=t")

# Model init
model = torch.load('./model.pt', map_location='cpu')
gradcam_model = torch.load('./model.pt', map_location='cpu')

# Grad cam init
target_layers = [gradcam_model.layer4[-1]]
cam = GradCAM(model=gradcam_model, target_layers=target_layers, use_cuda=False)

# Preprocess
transform = transforms.Compose(
    [
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])


def get_prediction(image, thres=0):
  global model
  model.eval()
  batch_size = 1
  trans_img = transform(image)
  trans_img = trans_img.view(1, 3, img_size[0], img_size[1])

  output = model(trans_img)
  output = nn.Softmax(dim=1)(output)
  output = output[0].tolist()
  output = [round(pred, 2) for pred in output]
  print(output)
  thres = 0.8
  final_decision = 0
  if output[0] < thres and output[1] < thres and output[2] < thres:
    final_decision = 3
  else:
    final_decision = output.index(max(output))

  result = {'ez': output[0], 'ps': output[1], 'others': output[2], 'normal': output[3],
            'final_decision': id2label[final_decision]}

  return result


img_size = (224, 224)

# Label information
label2id = {'ez': 0, 'ps': 1, 'others': 2, 'normal': 3}
id2label = {v: k for k, v in label2id.items()}


def download_model(url, output='model.pt'):
  gdown.download(url, output, quiet=False)


if not os.path.exists('./model.pt'):
  # Model v1
  #  download_model("https://drive.google.com/u/0/uc?id=1GH-YFFloEULAEU3bk55iLJkRV3z9hbKL&export=download&confirm=t")
  # Model v2
  download_model(
      "https://drive.google.com/u/0/uc?id=1ZpWItLNAxlMj0nNGcPldGpl6meV6rSbO&export=download&confirm=t")
# Model init
model = torch.load('./model.pt', map_location='cpu')
gradcam_model = torch.load('./model.pt', map_location='cpu')
# Grad cam init
target_layers = [gradcam_model.layer4[-1]]
cam = GradCAM(model=gradcam_model, target_layers=target_layers, use_cuda=False)
# Preprocess
transform = transforms.Compose(
    [
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])


def get_prediction(image, thres=0):
  global model
  model.eval()
  batch_size = 1
  trans_img = transform(image)
  trans_img = trans_img.view(1, 3, img_size[0], img_size[1])

  output = model(trans_img)
  output = nn.Softmax(dim=1)(output)
  output = output[0].tolist()
  output = [round(pred, 2) for pred in output]
  print(output)
  thres = 0.8
  final_decision = 0
  if output[0] < thres and output[1] < thres and output[2] < thres:
    final_decision = 3
  else:
    final_decision = output.index(max(output))

  result = {'ez': output[0], 'ps': output[1], 'others': output[2], 'normal': output[3],
            'final_decision': id2label[final_decision]}

  return result


def get_gradcam(image, class_name):

  batch_size = 1
  trans_img = transform(image)
  trans_img = trans_img.view(1, 3, img_size[0], img_size[1])

  class_index = label2id[class_name]
  print(class_index)
  targets = [ClassifierOutputTarget(class_index)]

  grayscale_cam = cam(input_tensor=trans_img, targets=targets)
  grayscale_cam = grayscale_cam[0, :]

  grayscale_cam_image = (grayscale_cam*255).astype(np.uint8)
  grayscale_cam_image = Image.fromarray(grayscale_cam_image, 'L')

  #base64_image = base64.b64encode(cv2.imencode('.jpg', grayscale_cam_image)[1]).decode()

  buffered = BytesIO()
  grayscale_cam_image.save(buffered, format="JPEG")
  base64_image = base64.b64encode(buffered.getvalue())
  result = {'heatmap': base64_image}

  return result
