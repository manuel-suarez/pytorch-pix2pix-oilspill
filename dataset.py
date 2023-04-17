import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

################## Augmentations ##############
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class Satellite2Map_Data(Dataset):
    def __init__(self, imgs_root, lbls_root):
        self.imgs_root = imgs_root
        self.lbls_root = lbls_root
        list_imgs_files = os.listdir(self.imgs_root)
        list_lbls_files = os.listdir(self.lbls_root)
        # The images were numerically (in the name) sorted
        list_imgs_files.sort()
        list_lbls_files.sort()
        #### Removing '.ipynb_checkpoints' from the list
        # list_files.remove('.ipynb_checkpoints')
        self.n_images = list_imgs_files
        self.n_labels = list_lbls_files

    def __len__(self):
        return len(self.n_images)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            # Open satellite image
            sat_image_name = self.n_images[idx]
            sat_image_path = os.path.join(self.imgs_root, sat_image_name)
            sat_image = np.asarray(Image.open(sat_image_path).convert('RGB'))
            # Open segmentation image
            map_image_name = self.n_labels[idx]
            map_image_path = os.path.join(self.lbls_root, map_image_name)
            map_image = np.asarray(Image.open(map_image_path).convert('RGB'))

            augmentations = both_transform(image=sat_image, image0=map_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

            sat_image = transform_only_input(image=input_image)["image"]
            map_image = transform_only_mask(image=target_image)["image"]
            return (sat_image, map_image)
        except:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_images[idx]
            image_path = os.path.join(self.imgs_root, image_name)
            print(image_path)
            pass

if __name__ == "__main__":
    dataset = Satellite2Map_Data("./dataset/train/images", "./dataset/train/labels")
    loader = DataLoader(dataset, batch_size=5)
    for x,y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, "satellite.png")
        save_image(y, "map.png")
        break