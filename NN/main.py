
#%%
# import pytorch and use fasterRCNN
import cv2
import time
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.ops.boxes import nms
import os

#%%
# the path and some values
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NUM_EPOCHS = 10
#%% md
# Loading Data
#%%
from StarfishDataSet import BASE_DIR

#read the csv file
df_train = pd.read_csv(BASE_DIR +"train.csv")
df_train.shape
#%%
df_train.head()
#%%
# Turn annotations from strings into lists of dictionaries
#delete the annotations which are empty
df_train=df_train.loc[df_train["annotations"].astype(str) != "[]"]
df_train['annotations'] = df_train['annotations'].apply(eval)
#Add image path
def image_path(r):
    video_id = r['video_id']
    video_frame = r['video_frame']
    return  "video_" + str(video_id) + "/" + str(video_frame) + ".jpg"
df_train['image_path'] = df_train.apply(lambda x: image_path(x), axis=1)

print(f'shape: {df_train.shape}')
df_train
#%%
# Check how many picture's annotations are not empty
(df_train['annotations'].str.len() > 0).value_counts()
#%%
# split the image into train and test
image_ids = df_train['image_id'].unique()
#80% for train and 20% for test.(%20*4919=983)
train_ids = image_ids[:-983]
test_ids = image_ids[-983:]
train_df = df_train[df_train['image_id'].isin(train_ids)]
test_df = df_train[df_train['image_id'].isin(test_ids)]

#%% md
# Dataset
#%%

#%%
# here we experiment image augmentation using simple transformations
# 50% probability of horizontal flip and 50% probability of vertical flip
# we resize all pictures as 720*1280
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=720, width=1280, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
#%%
from StarfishDataSet import StarfishDataset

ds_train = StarfishDataset(train_df, get_train_transform())
ds_val = StarfishDataset(test_df, get_valid_transform())
#%% md
# Check the sample
#%%
# From  train_df.shape we know there are 3936 sequences,we hope to check the last one picture
image, targets = ds_train[ds_train.__sizeof__() - 1]
#%%
# In image,show the starfish in the boxes
'''boxes = targets['boxes'].cpu().numpy().astype(np.int32)
img = image.permute(1,2,0).cpu().numpy()#to put the channels as the last dimension
fig, ax = plt.subplots(1, 1, figsize=(15, 7))

for box in boxes:
    cv2.rectangle(img,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)

ax.set_axis_off()
ax.imshow(img)'''
#%% md
# Create Model and load data to it
#%%

#%%
from collate import collate_fn

# Create PyTorch DataLoader
#load the data for the model

#Due to the train speed, we adjust batch_size=8 and num_worker=2
dl_train = DataLoader(ds_train, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
dl_val = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
#%%
def get_model():
    # load a resnet50 model and pre-train on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    # 1 class (starfish) + background
    num_classes = 2
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(DEVICE)
    return model

model = get_model()
#%%
# start to train and use torch.SGD to optimize
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

n_batches, n_batches_val = len(dl_train), len(dl_val)
validation_losses = []

print("start traing")
for epoch in range(NUM_EPOCHS):
    time_start = time.time()
    loss_accum = 0

    for batch_idx, (images, targets) in enumerate(dl_train, 1):

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Predict
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_accum += loss_value

        # Back-prop
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Validation
    val_loss_accum = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dl_val, 1):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            val_loss_dict = model(images, targets)
            val_batch_loss = sum(loss for loss in val_loss_dict.values())
            val_loss_accum += val_batch_loss.item()

    # Logging
    val_loss = val_loss_accum / n_batches_val
    train_loss = loss_accum / n_batches
    validation_losses.append(val_loss)

    # Save model
    chk_name = f'fasterrcnn_resnet50_fpn-e{epoch}.bin'
    torch.save(model.state_dict(), chk_name)


    elapsed = time.time() - time_start

    print(f"[Epoch {epoch+1:2d} / {NUM_EPOCHS:2d}] Train loss: {train_loss:.3f}. Val loss: {val_loss:.3f}  [{elapsed:.0f} secs]")

# find the minimum value loss
np.argmin(validation_losses)
# save the model after train
torch.save(model, "my_model.pth")