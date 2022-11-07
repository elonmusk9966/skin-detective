import torch
import torchvision.transforms as transforms
import torch.nn as nn

img_size = (224, 224)
label2id = {'ez':0, 'ps':1, 'others':2}
id2label = {v:k for k, v in label2id.items()}

model = torch.load('./best-model-fold-0.pt', map_location ='cpu')


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

    result = {'ez': output[0], 'ps': output[1], 'others': output[2]}

    return result

