import numpy as np
import torch
import torchvision
from tqdm import tqdm
#calulate the mean and std of the PCAM dataset
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.PCAM(root='./data', download=True, split='train', transform=transforms)

means = np.zeros(3)
stds = np.zeros(3)
#image is a pli image
for image,label in tqdm(dataset):
    #pil image to numpy
    image = np.array(image)
    for i in range(3):
        means[i] += image[i,:,:].mean()
        stds[i] += image[i,:,:].std()

means /= len(dataset)
stds /= len(dataset)

print("means: ", means)
print("stds: ", stds)



