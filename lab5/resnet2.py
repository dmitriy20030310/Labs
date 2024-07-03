import numpy as np
import torch
import torch.nn as F
import torchvision
import cv2
import matplotlib.pyplot as plt
transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

avgpool_features = None


def get_features(module, inputs, output):
    global avgpool_features
    avgpool_features = output


def get_features_map():
    return avgpool_features


img = cv2.cvtColor(cv2.imread('din1.jpg'), cv2.COLOR_BGR2RGB)
image = transform_pipe(img)

image2find = cv2.cvtColor(cv2.imread('uyoba.jpg'), cv2.COLOR_BGR2RGB)
image2find = transform_pipe(image2find)

image = image[None, :, :, :]
image2find = image2find[None, :, :, :]

model = torchvision.models.resnet50(pretrained=True)
model.layer4.register_forward_hook(get_features)

model.eval()

model.forward(image)
avg = get_features_map()

model.forward(image2find)
avg2find = get_features_map()

avg = F.functional.normalize(avg)
avg2find = F.functional.normalize(avg2find)

res = F.functional.conv2d(avg, avg2find)

res = res / (avg2find.shape[2]*avg2find.shape[3]) * 255

transform = torchvision.transforms.Resize(size=(image.shape[2], image.shape[3]))

res = transform(res)

res = res[0][0].cpu().detach().numpy()
res = res.astype(np.uint8)

plt.imshow(res)
plt.show()
