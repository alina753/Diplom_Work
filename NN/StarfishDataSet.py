import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
from torch.utils.data import Dataset

BASE_DIR = "D:/79381/Downloads/Diplom_Work/tensorflow-great-barrier-reef/"

# Define a map style Datasets
class StarfishDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
    # read the image,faster-rcnn model expects input to be in range [0-1]
    def get_image(self, row):
        image = cv2.imread(f'{BASE_DIR}train_images/{row["image_path"]}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Picture normalization
        image /= 255.0
        return image

    # Get the bounding boxes
    def get_boxes(self, row):
        boxes = pd.DataFrame(row['annotations'], columns=['x', 'y', 'width', 'height']).astype(float).values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # check if the bboxes are valid，image pixels are 1280*720
        boxes[:, 2] = np.clip(boxes[:, 2], 0, 1280)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, 720)
        return boxes

    def __getitem__(self, i):

        row = self.df.iloc[i]
        image = self.get_image(row)
        boxes = self.get_boxes(row)

        #count of boxes
        n_boxes = boxes.shape[0]

        # Calculate the area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Create a target dictionary
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'image_id': torch.tensor([i]),
            'labels': torch.ones((n_boxes,), dtype=torch.int64),
            'iscrowd': torch.zeros((n_boxes,), dtype=torch.int64)
        }

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transforms(**sample)
            image = sample['image']

            if n_boxes > 0:
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        else:
            image = ToTensorV2(p=1.0)(image=image)['image']

        return image, target

    def __len__(self):
        return len(self.df)
