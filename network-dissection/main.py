import torchvision.models
import settings
from loader.model_loader import loadmodel  #加载模型
from feature_operation import hook_feature,FeatureOperator   #特征提取
from visualize.report import generate_html_summary  #可视化
from util.clean import clean   #clean
import torch
from util.rotate import randomRotationPowers
import numpy as np
import time
import os

import torch
import torch.nn as nn
import os
# start = time.clock()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
import torch.nn as nn
print ('device is available:',torch.cuda.is_available())
print (torch.version.cuda)
print ('#device:', torch.cuda.device_count())
device = torch.device("cuda:0")
print ('device#:', torch.cuda.current_device())
print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))


fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############89
features, maxfeature = fo.feature_extraction(model=model)


for layer_id,layer in enumerate(settings.FEATURE_NAMES):
############ STEP 2: calculating threshold ############

    thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()

