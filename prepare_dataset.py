# The images in oil spill dataset have 1250x650 pixels, so we must divide them and adjust to
import os
from PIL import Image
from tqdm import tqdm
ORIG_DIR = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset"
DEST_DIR = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset_256"

def split_images(origin, dest, size, prefix, ext):
    os.makedirs(dest, exist_ok=True)
    list_files = os.listdir(origin)
    list_files.sort()
    # Use steps of 64x64
    step_x = size
    step_y = size
    for fname in tqdm(list_files):
        img_number = fname.split('.')[0].split('_')[1]
        im = Image.open(os.path.join(origin, fname))
        # Crop subimages
        counter = 1
        for i in range(1250 // size):
            for j in range(650 // size):
                dims = (i * step_x, j * step_y, i * step_x + size, j * step_y + size)
                ims = im.crop(dims)
                seq = "{:02d}".format(counter)
                ims.save(os.path.join(dest, f"{prefix}_{img_number}_{seq}.{ext}"))
                counter += 1

def split_dataset(origin, dest, dims):
    for setdir in ['train', 'test', 'val']:
        # Split images
        split_images(os.path.join(origin, setdir, 'images'), os.path.join(dest, setdir, 'images'), dims, 'img', 'jpg')
        # Split labels
        split_images(os.path.join(origin, setdir, 'labels'), os.path.join(dest, setdir, 'labels'), dims, 'img', 'png')
        # Split labels 1D
        split_images(os.path.join(origin, setdir, 'labels_1D'), os.path.join(dest, setdir, 'labels_1D'), dims, 'img', 'png')

split_dataset(ORIG_DIR, DEST_DIR, 256)
