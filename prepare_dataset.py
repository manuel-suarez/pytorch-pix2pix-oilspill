# The images in oil spill dataset have 1250x650 pixels so we must divide them and adjust to
import os
from config import TRAIN_DIR, TEST_DIR, VAL_DIR
from PIL import Image
from tqdm import tqdm

def split_images(origin, dest, prefix, ext):
    os.makedirs(dest)
    list_files = os.listdir(origin)
    list_files.sort()
    counter = 1
    step_x = (1200 - 256) // 9
    step_y = (600 - 256) // 4
    for fname in tqdm(list_files):
        im = Image.open(os.path.join(origin, fname))
        # Crop subimages
        for i in range(10):
            for j in range(5):
                dims = (i * step_x, j * step_y, i * step_x + 256, j * step_y + 256)
                print(dims)
                #ims = im.crop(dims)

        #im1 = im.crop((1, 1, 600, 600))
        #seq = "{:04d}".format(counter)
        #im1.save(os.path.join(dest, f"{prefix}_{seq}.{ext}"))
        #counter += 1
        # Right image
        #im2 = im.crop((601, 0, 1200, 600))
        #seq = "{:04d}".format(counter)
        #im2.save(os.path.join(dest, f"{prefix}_{seq}.{ext}"))
        #counter += 1

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