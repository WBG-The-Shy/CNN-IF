import settings
import torch
import torchvision
import settings
import torch
import torchvision
import torch.nn as nn
from src.torch_joint_training_sequences import *
from src.torch_gnet import Encoder
from src.torch_alexnet import AlexNet,alexnet_fmaps
from src.torch_alexnet_filter import AlexNet_filter,alexnet_fmaps_filter
from src.torch_resnet import ResNet50,Bottleneck
from src.torch_mpf import Torch_LayerwiseFWRF
from src.torch_vgg16 import VGG16,vgg_fmaps
from src.torch_vgg16_filter import VGG16_filter,vgg_fmaps_filter
import torchvision
import numpy as np

device = torch.device("cuda:0")
def loadmodel(hook_fn):

    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        # model._modules.get(name).register_forward_hook(hook_fn) #??forward????????????????hook???????? ???forward???????????
        # fea_hooks = []
        for n, m in model.named_modules():
            # if isinstance(m, torch.nn.Conv2d):
            #     m.register_forward_hook(hook_fn)
            # if isinstance(m, torch.nn.MaxPool2d):
            #     m.register_forward_hook(hook_fn)
            # if isinstance(m, torch.nn.ReLU):
            m.register_forward_hook(hook_fn)
            # fea_hooks.append(m)
        # print(len(fea_hooks))
    if settings.GPU:
        model.to(device)
    model.eval() #before using model to predict, it needs to close the batchnormal and droppout
    return model
