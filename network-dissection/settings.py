######### global settings  #########
import torch
import numpy as np
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result清理临时大文件
MODEL = 'AlexNet'                            # model arch: AlexNet, GNet, VGG16, ResNet50
trn_subjects = [1]
trn_str = '1'                     # select the training dataset and subject
ROI_name = "V1_V2_V3_V4_FFA_EBA_RSC_VWFA"
pretrained = True
feature_filter = True
learning_rate = 1e-3
session = 16
batch_size = 50
num_epochs = 50
holdout_size = 1200
n_resample = 100
training_order = "AlexNet_Jul-21-2023_1719"
normalization = False
ND_ROI = "RSC"
ND_SUB = 1
DATASET = 'imagenet'                       # model trained on: places365 or imagenet
QUANTILE = 0.01                            # the threshold used for activation
SEG_THRESHOLD = 0.01                       # the threshold used for visualization
SCORE_THRESHOLD = 0.04                     # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                               # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part", "scene", "material", "texture", "color"]
cate = 6
# concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
if pretrained:
    pre = "pretrained"
else:
    pre = "unpretrained"
if normalization:
    nor = "normal"
else:
    nor = "unnormal"
root_dir = ""  # get the encoding model
OUTPUT_FOLDER = "result" # result will be stored in this folder
########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'AlexNet':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if MODEL == 'GNet':
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227
if MODEL == 'resnet50':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000

if MODEL == 'AlexNet':
    FEATURE_NAMES = ['feature map 5']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/alexnet_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = "zoo/alexnet-owt-4df8aa71.pth"
        MODEL_PARALLEL = False
elif MODEL == 'resnet18':
    FEATURE_NAMES = ['layer3']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = 'zoo/resnet18-f37072fd.pth'
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/densenet161_places365.pth .tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = 'zoo/densenet161-8d451a50.pth'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer2']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet50_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = 'zoo/ResNet50_FFA_4session_50epoch_32bs_300hold.pth'
        MODEL_PARALLEL = False
elif MODEL == 'vgg16':
    FEATURE_NAMES = ['features']
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/vgg16_4session_50epoch_16bs_normal_lr=-5_meanstd.pth'
        MODEL_PARALLEL = False
elif MODEL == 'GNet':
    FEATURE_NAMES = ['pre']
    if DATASET == 'imagenet':
        MODEL_FILE = ""
# elif MODEL == 'alexnet':
#     FEATURE_NAMES = ['features']
#     if DATASET == 'places365':
#         MODEL_FILE = 'zoo/alexnet_places365.pth.tar'
#         MODEL_PARALLEL = False
#     elif DATASET == 'imagenet':
#         MODEL_FILE = 'zoo/alexnet-owt-7be5be79.pth'
#         MODEL_PARALLEL = False


if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4#64
    TALLY_BATCH_SIZE = 4#16
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 2024
    TALLY_BATCH_SIZE = 2024
    TALLY_AHEAD = 8
    INDEX_FILE = 'index.csv'
