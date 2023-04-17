# The images in oil spill dataset have 1250x650 pixels so we must divide them and adjust to
import os
import config

def split_images(directory):
    list_files = os.listdir(directory)
    for f in list_files:
        print(f)

# Split images
split_images(os.path.join(config.TRAIN_DIR, 'images'))
# Split labels