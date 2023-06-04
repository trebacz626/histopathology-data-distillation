import numpy as np
import torchvision
import torch
from torchvision.transforms import transforms
import time
from tqdm import tqdm
import os

dataset = torchvision.datasets.PCAM(root='./data', download = True, split='val')

#load PCAM target file
targets_file = dataset._base_folder / dataset._FILES[dataset._split]["targets"][0]


with dataset.h5py.File(targets_file) as targets_data:
    y = targets_data["y"][:,0, 0, 0]

y_1_idx = np.where(y == 1)[0]
y_0_idx = np.where(y == 0)[0]
raise "aa"

#save y_1_idx and y_0_idx in data/pcam/spllits create direcrories if they do not exist

if not os.path.exists('./data/pcam/splits'):
    os.makedirs('./data/pcam/splits')
np.save('./data/pcam/splits/y_1_idx.npy', y_1_idx)
np.save('./data/pcam/splits/y_0_idx.npy', y_0_idx)




#split the dataset into 2 classes
class PCAMClass(torchvision.datasets.PCAM):
    def __init__(self, root, split, class_mapping=None, transform=None, target_transform=None, download=False):
        super(PCAMClass, self).__init__(root, split, transform, target_transform, download)
        self.class_mapping = class_mapping
    def __len__(self):
        return len(self.class_mapping)
    def __getitem__(self, index):
        img, target = super(PCAMClass, self).__getitem__(self.class_mapping[index])
        return img, target

transform = transforms.Compose([transforms.ToTensor()])

dataset_0 = PCAMClass(root='./data', split='train', class_mapping=y_0_idx, transform=transform)
dataset_1 = PCAMClass(root='./data', split='val', class_mapping=y_1_idx, transform=transform)

print("dataset_0: ", len(dataset_0))
print("dataset_1: ", len(dataset_1))
print("dataset: ", len(dataset), "dataset_0 + dataset_1: ", len(dataset_0) + len(dataset_1))

#dataloaders
dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=256, shuffle=True, num_workers=1)
dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=256, shuffle=True, num_workers=1)

dataloader_0_iterator = iter(dataloader_0)
dataloader_1_iterator = iter(dataloader_1)
def get_image_batch(clas):
    global dataloader_0_iterator, dataloader_1_iterator
    if clas == 0:
        try:
            return next(dataloader_0_iterator)
        except StopIteration:
            dataloader_0_iterator = iter(dataloader_0)
            return next(dataloader_0_iterator)
    else:
        try:
            return next(dataloader_1_iterator)
        except StopIteration:
            dataloader_1_iterator = iter(dataloader_1)
            return next(dataloader_1_iterator)




def measure_time():
    start = time.time()
    for i in tqdm(range(200)):
        batch = get_image_batch(0)
    end = time.time()
    return end - start

print("Time to read a batch: ", measure_time())
