import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import skimage.io as io

class CustomDataset(Dataset):
    def __init__(self, filename, annFile, transform=None):
        self.file = h5py.File(filename, 'a')
        self.idxs = [int(key.split('_')[1]) for key in list(self.file.keys())]
        self.coco_kps=COCO(annFile)
        self.transform = transform

        cat_ids = self.coco_kps.getCatIds()
        self.cats = self.coco_kps.loadCats(cat_ids)

    def __len__(self):
        return len(self.idxs)

    def get_keypoints(self):
        return self.cats[0]['keypoints']

    def get_connections(self):
        return self.cats[0]['skeleton']

    def __getitem__(self, idx):
        img = self.coco_kps.loadImgs(self.idxs[idx])[0]
        I = Image.fromarray(io.imread(img['coco_url'])).convert("RGB")

        if self.transform:
            I = self.transform(I)
        I = np.array(I)

        parts = self.file[f'image_{self.idxs[idx]}/parts'][:]
        connections = self.file[f'image_{self.idxs[idx]}/connections'][:]
        return I, (parts, connections)
