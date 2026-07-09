import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMG_MEAN, IMG_STD


#CelebA dataset loading
CELEBA_ROOT = os.path.join("data", "celeba")
CELEBA_IMG_DIR = os.path.join(CELEBA_ROOT, "img_align_celeba", "img_align_celeba")


#CelebA smile/non-smile dataset from Kaggle CSV files
class CelebASmile(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(CELEBA_IMG_DIR, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_celeba():
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    #read attributes and partition CSVs
    attrs = pd.read_csv(os.path.join(CELEBA_ROOT, "list_attr_celeba.csv"))
    partitions = pd.read_csv(os.path.join(CELEBA_ROOT, "list_eval_partition.csv"))

    #merge on image_id
    df = attrs.merge(partitions, on="image_id")

    #smile labels: convert from {-1, 1} to {0, 1}
    df["label"] = (df["Smiling"] == 1).astype(int)

    #split by partition: 0=train, 1=val, 2=test
    train_df = df[df["partition"] == 0]
    test_df = df[df["partition"] == 2]

    train = CelebASmile(train_df["image_id"].tolist(),
                        train_df["label"].tolist(), transform)
    test = CelebASmile(test_df["image_id"].tolist(),
                       test_df["label"].tolist(), transform)

    print(f"  CelebA (smile/non-smile): {len(train)} train, {len(test)} test samples")
    return train, test


#IID partition: assign data_per_client random samples to each client
def partition_iid(dataset, num_clients, data_per_client):
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i][:data_per_client].tolist() for i in range(num_clients)}
