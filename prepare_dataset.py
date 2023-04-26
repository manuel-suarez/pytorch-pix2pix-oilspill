# The images in oil spill dataset have 1250x650 pixels, so we must divide them and adjust to
import os
from config import TRAIN_DIR, TEST_DIR, VAL_DIR
from PIL import Image
from tqdm import tqdm

def split_images(origin, dest, prefix, ext):
    os.makedirs(dest)
    list_files = os.listdir(origin)
    list_files.sort()
    # Use steps of 64x64
    step_x = 64 #(1200 - 256) // 3
    step_y = 64 #(600 - 256) // 1
    for fname in tqdm(list_files):
        img_number = fname.split('.')[0].split('_')[1]
        im = Image.open(os.path.join(origin, fname))
        # Crop subimages
        counter = 1
        for i in range(1200 // 64):
            for j in range(600 // 64):
                dims = (i * step_x, j * step_y, i * step_x + 64, j * step_y + 64)
                ims = im.crop(dims)
                seq = "{:02d}".format(counter)
                ims.save(os.path.join(dest, f"{prefix}_{img_number}_{seq}.{ext}"))
                counter += 1

def split_dataset(origin, dest):
    # Split images
    split_images(os.path.join(origin, 'images'), f"./dataset/{dest}/images", 'img', 'jpg')
    # Split labels
    split_images(os.path.join(origin, 'labels'), f"./dataset/{dest}/labels", 'img', 'png')

# Train dataset
split_dataset(TRAIN_DIR, 'train')
# Test dataset
split_dataset(TEST_DIR, 'test')
# Val dataset
split_dataset(VAL_DIR, 'val')