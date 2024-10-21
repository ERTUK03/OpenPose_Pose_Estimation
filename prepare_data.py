import skimage.io as io
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
import h5py
import torch
import os

def prepare_data(annFile, annSaveFile, o, size):
    if os.path.isfile(annSaveFile):
        print('Data already exists. Skipping preparation.')
        return

    coco_kps=COCO(annFile)

    cat_ids = coco_kps.getCatIds(catNms=['person'])
    cats = coco_kps.loadCats(cat_ids)

    keypoints = cats[0]['keypoints']
    connections = cats[0]['skeleton']

    resize_transform = transforms.Resize(size)

    conns = {}

    for i in range(len(keypoints)):
        conns[i] = []

    for c1, c2 in connections:
        conns[c1-1].append(c2-1)

    id_list = coco_kps.getImgIds(catIds=cat_ids)

    index = 0
    length = len(id_list)


    with h5py.File(annSaveFile, 'a') as f:
        for id in id_list:
            index=index+1

            img = coco_kps.loadImgs(id)[0]
            annot = coco_kps.loadAnns(coco_kps.getAnnIds(imgIds=id))

            I = Image.fromarray(io.imread(img['coco_url']))

            I = resize_transform(I)

            parts_maps = np.zeros((max(len(annot),1), len(keypoints), I.size[0], I.size[1]))
            connections_maps = np.zeros((max(len(annot),1), len(connections), I.size[0], I.size[1], 2))

            i_indices, j_indices = np.indices((I.size[0], I.size[1]))

            scale_w = I.size[1] / img['width']
            scale_h = I.size[0] / img['height']

            if annot:
                for i in range(len(annot)):
                    annot[i]['keypoints'] = torch.tensor(annot[i]['keypoints'])
                    annot[i]['keypoints'][::3] = torch.round(annot[i]['keypoints'][::3] * scale_w)
                    annot[i]['keypoints'][1::3] = torch.round(annot[i]['keypoints'][1::3] * scale_h)
                    annot[i]['keypoints'] = annot[i]['keypoints'].numpy()

            for i, obj in enumerate(annot):
                iter=0
                for j, keypoint in enumerate(keypoints):
                    if obj['keypoints'][j*3+2]!=2:
                        iter+=len(conns[j])
                        continue

                    parts_maps[i][j] = np.exp(-((np.sqrt((i_indices - obj['keypoints'][j*3+1]) ** 2 + (j_indices - obj['keypoints'][j*3]) ** 2))/o**2))

                    for con in conns[j]:
                        if obj['keypoints'][con*3+2]!=2:
                            iter+=1
                            continue

                        distance=np.sqrt((obj['keypoints'][con*3+1]-obj['keypoints'][j*3+1])**2+(obj['keypoints'][con*3]-obj['keypoints'][j*3])**2)

                        if distance == 0:
                            continue

                        v=((obj['keypoints'][con*3+1]-obj['keypoints'][j*3+1])/distance,(obj['keypoints'][con*3]-obj['keypoints'][j*3])/distance)

                        σ = 0.1*distance

                        v_t = (-v[1], v[0])

                        cx, cy = np.meshgrid(np.arange(I.size[0]), np.arange(I.size[1]), indexing='ij')
                        x_diff = cx - obj['keypoints'][j*3+1]
                        y_diff = cy - obj['keypoints'][j*3]

                        diff_vectors = np.stack([x_diff, y_diff], axis=-1)

                        dot_products = np.dot(diff_vectors, v)
                        dot_products_2 = np.dot(diff_vectors, v_t)

                        mask = (dot_products >= 0) & (dot_products < distance) & (np.abs(dot_products_2) <= σ)

                        connections_maps[i][iter] = np.where(mask[..., None], v, (0,0))

                        iter+=1

            max_parts_maps = np.maximum.reduce(parts_maps)

            max_connections_count = np.any(connections_maps != 0, axis=-1).astype(int)
            max_connections_count = np.sum(max_connections_count, axis=0)
            max_connections_count = max_connections_count[...,np.newaxis]

            max_connections_maps = np.sum(connections_maps, axis=0)

            max_connections_maps = np.divide(max_connections_maps, max_connections_count, where=(max_connections_count != 0), out = max_connections_maps)

            f.create_dataset(f'image_{id}/parts', data=max_parts_maps, compression='gzip')
            f.create_dataset(f'image_{id}/connections', data=max_connections_maps, compression='gzip')

            print(f"Processed image {index}/{length}")
