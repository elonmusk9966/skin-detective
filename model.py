import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gdown
import os

img_size = (224, 224)

# Label information
label2id = {'ez':0, 'ps':1, 'others':2, 'normal':3}
id2label = {v:k for k, v in label2id.items()}

def download_model(url, output='model.pt'):
    gdown.download(url, output, quiet=False)


if not os.path.exists('./model.pt'):
    ## Model v1
    #  download_model("https://drive.google.com/u/0/uc?id=1GH-YFFloEULAEU3bk55iLJkRV3z9hbKL&export=download&confirm=t")
    ## Model v2
    download_model("https://drive.google.com/u/0/uc?id=1ZpWItLNAxlMj0nNGcPldGpl6meV6rSbO&export=download&confirm=t")

model = torch.load('./model.pt', map_location ='cpu')

def get_prediction(image, thres=0):
    global  model
    model.eval()
    transform = transforms.Compose(
        [
          transforms.Resize(size=img_size),
          transforms.ToTensor(),
          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
          ])
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
