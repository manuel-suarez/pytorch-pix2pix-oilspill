# The images in oil spill dataset have 1250x650 pixels so we must divide them and adjust to
import os
import config
from PIL import Image
from tqdm import tqdm

def split_images(origin, dest, prefix, ext):
    os.makedirs(dest)
    list_files = os.listdir(origin)
    list_files.sort()
    counter = 1
    for fname in tqdm(list_files):
        im = Image.open(os.path.join(origin, fname))
        # Left image
        im1 = im.crop((1, 1, 600, 600))
        seq = "{:04d}".format(counter)
        im1.save(os.path.join(dest, f"{prefix}_{seq}.{ext}"))
        counter += 1
        # Right image
        im2 = im.crop((601, 0, 1200, 600))
        seq = "{:04d}".format(counter)
        im2.save(os.path.join(dest, f"{prefix}_{seq}.{ext}"))
        counter += 1

# Split images
split_images(os.path.join(config.TRAIN_DIR, 'images'), './dataset/train/images', 'img', 'jpg')
# Split labels
split_images(os.path.join(config.TRAIN_DIR, 'labels'), './dataset/train/labels', 'img', 'png')