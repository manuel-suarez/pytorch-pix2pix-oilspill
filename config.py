import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset/train"
TEST_DIR = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset/test"
VAL_DIR = "/home/est_posgrado_manuel.suarez/data/oil-spill-dataset/val"
LEARNING_RATE = 2e-4
BETA1 = 0.5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
