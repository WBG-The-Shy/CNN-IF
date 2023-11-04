from IPython.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))

import sys
import os
import struct
import time
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import ResNet50_Weights
from torchvision.models import VGG16_Weights
import torchvision

fpX = np.float32
import src.numpy_utility as pnu
from src.file_utility import save_stuff, flatten_dict, embed_dict

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
from PIL import Image
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # locate gpu matter
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
import torch

print('#device:', torch.cuda.device_count())
print('device#:', torch.cuda.current_device())
print('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

torch.manual_seed(time.time())
torch.backends.cudnn.enabled = True
#
# print('\ntorch:', torch.__version__)
# print('cuda: ', torch.version.cuda)
# print('cudnn:', torch.backends.cudnn.version())
# print('dtype:', torch.get_default_dtype())

sns.axes_style()
sns.set_style("whitegrid", {"axes.facecolor": '.95'})
sns.set_context("notebook",
                rc={'axes.labelsize': 18.0, 'axes.titlesize': 24.0, 'legend.fontsize': 18.0, 'xtick.labelsize': 18.0,
                    'ytick.labelsize': 18.0})
sns.set_palette("deep")

device = torch.device("cuda:2")  # cuda
trn_subjects = [1,2,3,5]
trn_str = '1,2,3,5'                     # select the training dataset and subject
model_name = 'GNet'                    # 'model_name_mpf_general'
ROI_name = "V1_V2_V3_V4_FFA_EBA_RSC_VWFA"
pretrained = True
quick_load = True
feature_filter = True
feature_filter_num = 512
feature_filter_size = 128
learning_rate = 1e-3
session = 4
batch_size = 50
num_epochs = 50
holdout_size = 300
order_dir = "GNet_Jul-18-2023_1348"
n_resample = 100
root_dir = os.getcwd() + '/'
net_dir = root_dir + "net/"
voxel_data_dir = "/code/voxel_data/" + ROI_name + '/' + trn_str + '/%dsession/voxel_data_subj:[%s]_upto:%d.h5py'%(session,trn_str,session)
filter_dir = "/code/" + 'feature filter/%s/%s/%d/' % (model_name, trn_str, feature_filter_num)
if pretrained:
    pre = "pretrained"
else:
    pre = "unpretrained"

saveext = ".png"
savearg = {'format': 'png', 'dpi': 120, 'facecolor': None}
timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())  # 'Aug-10-2020_1638' #

model_dir = '%s_%s' % (model_name, timestamp)

output_dir = "/home/mufan/VDisk1/Mufan/Training_Result/drawimage/%s/%s/%s/%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d/%s_%s/"%(
    model_name,ROI_name,pre,session,trn_str,num_epochs,batch_size,learning_rate,holdout_size,model_name, timestamp)
video_dir = root_dir + "video/"
print(output_dir)
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(net_dir):
    os.makedirs(net_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for k, s in enumerate(trn_subjects):
    subject_dir = output_dir + 'S%02d/' % s
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
# print("Time Stamp: %s" % timestamp)
f = open(output_dir + '/settings.txt', "w+")
f.write("training subjects =  %s\n"%trn_str)
f.write("model name =  %s\n"%model_name)
f.write("ROI name =  %s\n"%ROI_name)
f.write("pre trained =  %s\n"%pre)
f.write("quick load =  %s\n"%quick_load)
f.write("feature filter =  %s\n"%feature_filter)
f.write("feature filter num =  %d\n"%feature_filter_num)
f.write("feature_filter_size = %d\n"%feature_filter_size)
f.write("learning_rate =  %.4f\n"%learning_rate)
f.write("session =  %d\n"%session)
f.write("batchsize =  %d\n"%batch_size)
f.write("num epochs =  %d\n"%num_epochs)
f.write("hold out size =  %d\n"%holdout_size)
f.write("n_resample =  %s\n"%n_resample)
f.write("output_dir =  %s\n"%output_dir)
f.close()
nsd_root = "/nsd/"
stim_root = nsd_root + "nsddata_stimuli/stimuli/nsd/"
beta_root = nsd_root + "nsddata_betas/ppdata/"
mask_root = nsd_root + "nsddata/ppdata/"

exp_design_file = nsd_root + "nsddata/experiments/nsd/nsd_expdesign.mat"

exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1  # zero-indexed ordering of indices (matlab-like to python-like)

image_data = {}
for s in trn_subjects:
    image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py" % s, 'r')
    image_data[s] = np.copy(image_data_set['stimuli'])
    image_data_set.close()
    print('--------  subject %d  -------' % s)
    print('block size:', image_data[s].shape, ', dtype:', image_data[s].dtype, ', value range:', \
          np.min(image_data[s][0]), np.max(image_data[s][0]))


from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi

from mask_preparation.mask import FFA
from mask_preparation.mask import V1_V2_V3_V4_FFA
from mask_preparation.mask import FFA_EBA_RSC_VWFA
from mask_preparation.mask import V1_V2_V3_V4
from mask_preparation.mask import V1_V2_V3_V4_FFA_EBA_RSC_VWFA

brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

if ROI_name == "FFA":
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = FFA(trn_subjects,mask_root,beta_root)
elif ROI_name == "V1_V2_V3_V4_FFA":
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = V1_V2_V3_V4_FFA(trn_subjects, mask_root, beta_root)
elif ROI_name == "FFA_EBA_RSC_VWFA":
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = FFA_EBA_RSC_VWFA(trn_subjects, mask_root, beta_root)
elif ROI_name == "V1_V2_V3_V4":
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = V1_V2_V3_V4(trn_subjects, mask_root, beta_root)
elif ROI_name == "V1_V2_V3_V4_FFA_EBA_RSC_VWFA":
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = V1_V2_V3_V4_FFA_EBA_RSC_VWFA(trn_subjects, mask_root, beta_root)

slice_idx = 35
plt.figure(figsize=(12, 4 * len(trn_subjects)))
for k, s in enumerate(trn_subjects):
    subject_dir = output_dir + 'S%02d/' % s
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    volume_brain_mask = view_data(brain_nii_shape[s], voxel_idx[s], np.ones_like(voxel_idx[s]),
                                  save_to=subject_dir + "subj%02d_mask" % s)
    volume_brain_roi = view_data(brain_nii_shape[s], voxel_idx[s], voxel_roi[s],
                                 save_to=subject_dir + "subj%02d_roi" % s)
    volume_ncsnr      = view_data(brain_nii_shape[s], voxel_idx[s], voxel_ncsnr[s], save_to=subject_dir+"subj%02d_ncsnr"%s)
    ##
    plt.subplot(len(trn_subjects), 3, 3 * k + 1)
    plt.imshow(volume_brain_mask[:, :, slice_idx], cmap='gray', interpolation='None')
    plt.title('Mask')
    plt.colorbar()
    _ = plt.ylabel('Subject %d' % s)
    _ = plt.gca().set_xticklabels([])
    _ = plt.gca().set_yticklabels([])

    plt.subplot(len(trn_subjects), 3, 3 * k + 2)
    plt.imshow(volume_brain_roi[:, :, slice_idx], cmap='jet', interpolation='None')
    plt.clim([0, 7])
    plt.title('ROI')
    plt.colorbar()
    _ = plt.gca().set_xticklabels([])
    _ = plt.gca().set_yticklabels([])

    plt.subplot(len(trn_subjects), 3, 3 * k + 3)
    plt.imshow(volume_ncsnr[:, :, slice_idx], cmap='Reds', interpolation='None')
    plt.title('NCSNR')
    plt.colorbar()
    _ = plt.ylabel('Subject %d' % s)
    _ = plt.gca().set_xticklabels([])
    _ = plt.gca().set_yticklabels([])

    plt.savefig(subject_dir + '%s_subj%02d_roi.jpg' % (ROI_name, s), dpi=800)
    plt.show()

# In[13]:

## Long version
from src.load_nsd import load_betas
from src.load_nsd import image_feature_fn, data_split

# quick load
if quick_load:
    voxel_data_set = h5py.File(voxel_data_dir,'r')
# voxel_data_set = h5py.File(root_dir+'voxel_data_general.h5py', 'r')
    voxel_data_dict = embed_dict({k: np.copy(d) for k, d in voxel_data_set.items()})
    voxel_data_set.close()
# %%
    voxel_data = voxel_data_dict['voxel_data']
    print(voxel_data.keys())
    data_size, nnv = {}, {}
    trn_stim_data, trn_voxel_data = {}, {}
    val_stim_single_trial_data, val_voxel_single_trial_data = {}, {}
    val_stim_multi_trial_data, val_voxel_multi_trial_data = {}, {}

    for k, s in enumerate(trn_subjects):
        data_size[s], nnv[s] = voxel_data.get(str(s)).shape
        print('--------  subject %d  -------' % s)
        trn_stim_data[s], trn_voxel_data[s], \
            val_stim_single_trial_data[s], val_voxel_single_trial_data[s], \
            val_stim_multi_trial_data[s], val_voxel_multi_trial_data[s] = \
            data_split(image_feature_fn(image_data[s]), voxel_data.get(str(s)), ordering,
                       imagewise=False)  # here voxel_data keys is str type with quick load

else:
    voxel_data = {}
    for k,s in enumerate(trn_subjects):
        print ('--------  subject %d  -------' % s)
        beta_subj = beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/"%s
        voxel_data[s],_ = load_betas(folder_name=beta_subj, zscore=True, voxel_mask=voxel_mask[s], up_to=session)
        print ('----------------------------')
        print (voxel_data[s].shape)
    voxel_dir = "/code/voxel_data/%s/%s/%dsession/"%(ROI_name,trn_str,session)
    if not os.path.exists(voxel_dir):
        os.makedirs(voxel_dir)
    # up_to = 1
    # root_dir + 'voxel_data/%s/%s/%dsession/voxel_data_subj:%s_upto_:%d'%(ROI_name,trn_str,session, trn_str,session)
    save_stuff(voxel_dir + '/voxel_data_subj:[%s]_upto:%d'%(trn_str,session),
        flatten_dict({
       'voxel_mask': voxel_mask,
       'voxel_roi': voxel_roi,
       'voxel_idx': voxel_idx,
       'voxel_ncsnr': voxel_ncsnr,
       'voxel_data': voxel_data
                }))
    data_size, nnv = {}, {}
    trn_stim_data, trn_voxel_data = {}, {}
    val_stim_single_trial_data, val_voxel_single_trial_data = {}, {}
    val_stim_multi_trial_data, val_voxel_multi_trial_data = {}, {}

    for k, s in enumerate(trn_subjects):
        data_size[s], nnv[s] = voxel_data.get(s).shape
        print('--------  subject %d  -------' % s)
        trn_stim_data[s], trn_voxel_data[s], \
            val_stim_single_trial_data[s], val_voxel_single_trial_data[s], \
            val_stim_multi_trial_data[s], val_voxel_multi_trial_data[s] = \
            data_split(image_feature_fn(image_data[s]), voxel_data.get(s), ordering,
                       imagewise=False)  # here voxel_data keys is str type with quick load
# del voxel_data
# del image_data

trn_stim_mean = sum([np.mean(trn_stim_data[s], axis=(0, 2, 3), keepdims=True) for s in trn_subjects]) / len(
    trn_subjects)
trn_stim_std = sum([np.std(trn_stim_data[s], axis=(0, 2, 3), keepdims=True) for s in trn_subjects]) / len(trn_subjects)
print('trn mean = ', trn_stim_mean)
print('trn std = ', trn_stim_std)

import src.torch_mpf as aaa
from importlib import reload

reload(aaa)

from src.torch_joint_training_sequences import *
from src.torch_gnet import Encoder
from src.torch_alexnet import AlexNet
from src.torch_alexnet_filter import AlexNet_filter,alexnet_fmaps_filter
from src.torch_resnet import ResNet50,Bottleneck
from src.torch_mpf import Torch_LayerwiseFWRF
from src.torch_vgg16 import VGG16,vgg_fmaps
from src.torch_vgg16_filter import VGG16_filter,vgg_fmaps_filter

_log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)

gent_paras = torch.load("/Training_Result/%s/%s/%s/%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d/%s/model_params"%(
    model_name,ROI_name,pre,session,trn_str,num_epochs,batch_size,learning_rate,holdout_size,order_dir))
num_epochs = gent_paras['num_epochs']
batch_size = gent_paras['batch_size']
holdout_size = gent_paras['holdout_size']
best_params = gent_paras['best_params']
final_params = gent_paras['final_params']
hold_hist = gent_paras['hold_loss_history']
trn_hist = gent_paras['trn_loss_history']
trn_cc_hist = gent_paras['trn_cc_history']
hold_cc_hist = gent_paras['hold_cc_history']
best_epoch = gent_paras['best_epoch']
best_joint_cc_score = gent_paras['best_joint_cc_score']
subject_val_cc = gent_paras['val_cc']
trn_stim_mean = gent_paras['input_mean']
trn_stim_std = gent_paras['input_std']
brain_nii_shape = gent_paras['brain_nii_shape']
voxel_idx = gent_paras['voxel_index']
voxel_roi = gent_paras['voxel_roi']
voxel_mask = gent_paras['voxel_mask']
voxel_ncsnr = gent_paras['voxel_ncsnr']
GNet = Encoder(trn_stim_mean, trunk_width=64).to(device)
GNet.load_state_dict(best_params['enc'])
rec, fmaps, h = GNet(T.from_numpy(trn_stim_data[trn_subjects[0]][:1]).to(device))
subject_fwrfs = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv[s], pre_nl=_log_act_fn, \
                                        post_nl=_log_act_fn, dtype=np.float32).to(device) for s in trn_subjects}
for i in trn_subjects:
    checkpoint = best_params['fwrfs'][i]
    subject_fwrfs[i].load_state_dict(checkpoint)

def _model_fn(_ext, _con, _x):
    '''model consists of an extractor (_ext) and a connection model (_con)'''
    x, _fm, h = _ext(_x)
    return _con(_fm)

def _smoothness_loss_fn(_rf, n):
    delta_x = T.sum(T.pow(T.abs(_rf[:, 1:] - _rf[:, :-1]), n))
    delta_y = T.sum(T.pow(T.abs(_rf[:, :, 1:] - _rf[:, :, :-1]), n))
    return delta_x + delta_y


def _loss_fn(_ext, _con, _x, _v):
    _r = _model_fn(_ext, _con, _x)
    _err = T.sum((_r - _v) ** 2, dim=0)
    _loss = T.sum(_err)
    _loss += fpX(1e-1) * T.sum(T.abs(_con.w))
    return _err, _loss


def _training_fn(_ext, _con, _opts, xb, yb):
    for _opt in _opts:
        _opt.zero_grad()
        _err, _loss = _loss_fn(_ext, _con, T.from_numpy(xb).to(device), T.from_numpy(yb).to(device))
        _loss.backward()
        _opt.step()
    return _err


def _holdout_fn(_ext, _con, xb, yb):
    _err, _ = _loss_fn(_ext, _con, T.from_numpy(xb).to(device), T.from_numpy(yb).to(device))
    return _err


def _pred_fn(_ext, _con, xb):
    return _model_fn(_ext, _con, T.from_numpy(xb).to(device))


def print_grads(_ext, _con, _params, _opt, xb, yb):
    _opt.zero_grad()
    _err, _loss = _loss_fn(_ext, _con, T.from_numpy(xb).to(device), T.from_numpy(yb).to(device))
    _loss.backward()
    for p in _params:
        prg = get_value(p.grad)
        print("%-16s : value=%f, grad=%f" % (list(p.size()), np.mean(np.abs(get_value(p))), np.mean(np.abs(prg))))
    print('--------------------------------------')
    sys.stdout.flush()


# Results

In[30]:
plt.figure(figsize=(30, 6))
plt.plot(np.array(trn_hist), color='r', marker='o', ms=10, lw=2)
plt.plot(np.array(hold_hist), color='b', marker='o', ms=10, lw=2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.title('Loss Curve')
plt.savefig(output_dir + 'loss.jpg', dpi=800)
plt.show()

# In[29]:

plt.figure(figsize=(30, 10))
plt.subplot(2, 1, 1)
for s, cc in trn_cc_hist.items():
    plt.gca().violinplot(np.nan_to_num(np.array(cc)).T, showmeans=True, showmedians=True, showextrema=True)
plt.ylabel('training history')
plt.title('training history')
plt.subplot(2, 1, 2)
for s in trn_cc_hist.keys():
    p = plt.plot([np.median(cc) for cc in trn_cc_hist[s]], lw=2, linestyle='--')
plt.title('Median history')
plt.tight_layout()
plt.legend(labels=['accuracy'])
plt.ylabel('Median history')
plt.xlabel('epoch')
plt.savefig(output_dir + 'train&mean_history.jpg', dpi=800)
plt.show()

plt.figure(figsize=(30, 10))
plt.subplot(2, 1, 1)
for s, cc in hold_cc_hist.items():
    plt.gca().violinplot(np.nan_to_num(np.array(cc)).T, showmeans=True, showmedians=True, showextrema=True)
plt.ylabel('Validation history')
plt.title('Validation history')
plt.subplot(2, 1, 2)
for s in hold_cc_hist.keys():
    p = plt.plot([np.median(cc) for cc in hold_cc_hist[s]], lw=2, linestyle='--')
plt.title('Median history')
plt.tight_layout()
plt.legend(labels=['accuracy'])
plt.ylabel('Median history')
plt.xlabel('epoch')
plt.savefig(output_dir + 'val&mean_history.jpg', dpi=800)
plt.show()
---
## Final validation accuracy
for k, s in enumerate(trn_subjects):
    a1 = val_stim_single_trial_data[s]
    b1 = torch.from_numpy(a1)
    c1 = transform.RandomRotation((90, 90), expand=False)(b1)
    d1 = c1.numpy()
    a2 = val_stim_single_trial_data[s]
    b2 = torch.from_numpy(a1)
    c2 = transform.RandomRotation((180,180), expand=False)(b1)
    d2 = c1.numpy()
    a3 = val_stim_single_trial_data[s]
    b3 = torch.from_numpy(a1)
    c3 = transform.RandomRotation((270,270), expand=False)(b1)
    d3 = c1.numpy()
    val_stim_single_trial_data[s] = np.concatenate([val_stim_single_trial_data[s],d1,d2,d3],axis=0)
    voxel_temp = val_voxel_single_trial_data[s]
    val_voxel_single_trial_data[s] = np.concatenate([val_voxel_single_trial_data[s],voxel_temp,voxel_temp,voxel_temp],axis=0)
val_voxel = {s: val_voxel_single_trial_data[s] for s in val_voxel_single_trial_data.keys()}

GNet.eval()
for s, sd in subject_fwrfs.items():
    sd.load_state_dict(best_params['fwrfs'][s])
    sd.eval()

subject_val_cc = validation_(_pred_fn, GNet, subject_fwrfs, val_stim_single_trial_data, val_voxel, batch_size) # all subject all voxels response to all images correlation
joined_val_cc = np.concatenate(list(subject_val_cc.values()), axis=0)
# compare = pre_val_cc - joined_val_cc
# plt.hist(compare,bins=1000,color='pink',edgecolor='b')
# plt.show()
volume_cc = {}
for k, s in enumerate(trn_subjects):
    subject_dir = output_dir + 'S%02d/' % s
    volume_cc[s] = view_data(brain_nii_shape[s], voxel_idx[s], subject_val_cc[s], save_to=subject_dir + "val_cc_%d"%s)
#
np.savetxt("/Training_Result/GNet_data_subj01_val.txt", subject_val_cc[1], fmt="%s", delimiter=",")
np.savetxt("/Training_Result/GNet_data_subj02_val.txt", subject_val_cc[2], fmt="%s", delimiter=",")
np.savetxt("/Training_Result/GNet_data_subj03_val.txt", subject_val_cc[3], fmt="%s", delimiter=",")
np.savetxt("/Training_Result/GNet_data_subj05_val.txt", subject_val_cc[5], fmt="%s", delimiter=",")
# In[45]:
import pandas as pd
import csv
from scipy import stats
import numpy as np
import torch as t
import matplotlib.pyplot as plt
for k, s in enumerate(trn_subjects):
    plt.figure()
    subject_dir = output_dir + 'S%02d/' % s
    ncsnr = voxel_ncsnr[s]
    val = subject_val_cc[s]
    plt.scatter(ncsnr, val, s=3, c='steelblue',label="individual voxels")
    slope, intercept, r_value, p_value, std_err = stats.linregress(ncsnr, val)
    mediam_x = np.median(ncsnr)
    median_y = np.median(val)
    X1 = ncsnr
    Y1 = np.array([intercept + slope * x for x in X1])
    plt.plot(X1, Y1, color='firebrick',label="Noise Ceiling")
    plt.scatter(mediam_x, median_y, s=50, c='firebrick', marker='+',label="Median across voxels")
    plt.xlabel("Noise Ceiling")
    plt.ylabel('Validation accuracy(R)')
    plt.legend(loc='lower right',fontsize=10)
    plt.title('S%d-%s'%(s,ROI_name))
    plt.tight_layout()
    plt.savefig(subject_dir + 'val_ncsnr_subj%02d_roi.jpg' %s, dpi=800)
    plt.show()



f = open(output_dir + 'val_acc.txt', "w+")
f.write("Run on subjects %s\n"%trn_str)


print("best training joint score = %.3f\n" % best_joint_cc_score)
print("median joint val cc = %.3f" % np.median(joined_val_cc))  # the joint validation accuracy median
f.write("best joint score = %.3f\n" % best_joint_cc_score) # the best joint cc score
f.write("median joint val cc = %.3f\n" % np.median(joined_val_cc)) # average among four subjects
for s, v in subject_val_cc.items():
    print("subject %s: val cc = %.3f" % (s, np.median(v)))  # the individual validation accuracy median
    f.write("subject %s: median val cc = %.3f\n" % (s, np.median(v)))
    f.write("subject %s: mean val cc = %.3f\n" % (s, np.mean(v)))
    f.write("subject %s: max val cc = %.3f\n" % (s, max(v)))
    f.write("subject %s: min val cc = %.3f\n" % (s, min(v)))
fig = plt.figure(figsize=(40, 10))
plt.subplots_adjust(left=0.1, bottom=0.2, right=1., top=1., wspace=0., hspace=0.)
_ = plt.hist(joined_val_cc, bins=100, density=True, range=(-.5, 1.))
_ = plt.vlines(x=[0], ymin=1e-4, ymax=25, color='r')
_ = plt.yscale('log')
_ = plt.xlim([-.1, 0.9])
_ = plt.xlabel(r'$\rho$')
_ = plt.ylabel('Relative frequency')
plt.savefig(output_dir + 'relative_frequency.jpg', dpi=800)
plt.show()
f.close()
plt.hist(joined_val_cc,bins=1000,color='pink',edgecolor='b')
plt.savefig(output_dir + 'frequency_count.jpg', dpi=800)
plt.show()

f = open(output_dir + 'model_notes.txt', "w+")
f.write("Run on subjects %s\n"%trn_str)
f.close()

subject_resample_val_cc = {}
for s in val_voxel_single_trial_data.keys():
    print('sampling subject %d' % s)
    subject_resample_val_cc[s] = cc_resampling_with_replacement(_pred_fn, GNet, subject_fwrfs[s],
                                                                val_stim_single_trial_data[s],
                                                                val_voxel_single_trial_data[s], batch_size, n_resample)

f = open(output_dir + 'model_notes.txt', "a")
f.write("-- median_validation_accuracies --\n")
fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=0.2, bottom=0.60, right=.95, top=.95, wspace=0., hspace=0.)
val_score_samples = [np.median(np.concatenate([ccs[i] for s, ccs in subject_resample_val_cc.items()], axis=0)) \
                     for i in range(n_resample)]
print('score mean = %.04f, std.dev = %.04f' % (np.mean(val_score_samples), np.std(val_score_samples)))
f.write('score mean = %.04f, std.dev = %.04f\n' % (np.mean(val_score_samples), np.std(val_score_samples)))
plt.plot([0, ] * len(val_score_samples), val_score_samples, linestyle='None', marker='o', ms=10)
plt.errorbar(x=[0, ], y=[np.mean(val_score_samples), ], yerr=[np.std(val_score_samples), ], marker='o', \
             color='k', ms=10, elinewidth=4, capsize=16, capthick=4)
xticks = [i for i in range(0,9)]
plt.xticks(xticks)
for k, (s, ccs) in enumerate(subject_resample_val_cc.items()):  # this is subject-wise result
    val_cc = np.zeros(ccs[0].shape,dtype=np.float32)
    for i in range(len(val_cc)):
        for cc in ccs:
            val_cc[i] = val_cc[i] + cc[i]
        val_cc[i] = val_cc[i] / n_resample
    subject_dir = output_dir + 'S%02d/' % s
    volume_cc[s] = view_data(brain_nii_shape[s], voxel_idx[s], subject_val_cc[s], save_to=subject_dir + "val_cc_%d" % s)
    mcc = [np.median(cc) for cc in ccs]
    print('Subject %s median cc mean = %.04f, std.dev = %.04f' % (s, np.mean(mcc), np.std(mcc)))
    f.write('Subject %s median cc mean = %.04f, std.dev = %.04f\n' % (s, np.mean(mcc), np.std(mcc)))
    plt.plot([s , ] * len(mcc), mcc, linestyle='None', marker='o', ms=10)
    plt.errorbar(x=[s , ], y=[np.mean(mcc), ], yerr=[np.std(mcc), ], marker='o', \
                 color='k', ms=10, elinewidth=4, capsize=16, capthick=4)
plt.ylabel('Median score')
plt.xlabel('validation subjects')
plt.savefig(output_dir + 'Median_score.jpg', dpi=800)
plt.show()
f.close()

# In[39]:

filename = output_dir + 'median_validation_accuracies%s' % (saveext)
fig.patch.set_alpha(0.)
fig.savefig(filename, **savearg)
plt.close()

# # Subject-wise analysis

# In[42]:
for i in trn_subjects:
    subj = i
    subject_dir = output_dir + 'S%02d/' % subj
    # In[43]:
    from matplotlib import cm


    def pooling_fn(x):  # np.exp(x)   e^x
        return np.exp(x) / np.sum(np.exp(x), axis=(1, 2), keepdims=True)


    vidxes = np.argsort(subject_val_cc[subj])  # return the order index from min to max
    vox = vidxes[-36:]  # np.arange(49) # get the max 36th voxel accuracy index
    n_x = int(np.floor(np.sqrt(len(vox))))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
    n_y = int(np.ceil(len(vox) / n_x)) + 1  # 计算大于等于改值的最小整数
    fig2a = plt.figure(figsize=(2 * n_x, 2 * n_y))
    for k, v in enumerate(vox):
        plt.subplot(n_y, n_x, k + 1)
        plt.imshow(pooling_fn(get_value(subject_fwrfs[subj].rfs[0])[v, np.newaxis])[0], interpolation='None',
                   cmap=cm.coolwarm, origin='upper')  # vidxes[-100:]
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        # plt.title(r'$\rho=$%.2f' % subject_val_cc[subj][v], fontsize=16)
    # In[44]:
    filename = output_dir + 'S%02d/subj%02d_rf0_sample%s' % (subj, subj, saveext)
    fig2a.patch.set_alpha(0.)    # set the transparent
    fig2a.savefig(filename, **savearg)
    plt.show()
    plt.close()
    # In[45]:
    fig2b = plt.figure(figsize=(2 * n_x, 2 * n_y))
    for k, v in enumerate(vox):
        plt.subplot(n_y, n_x, k + 1)
        plt.imshow(pooling_fn(get_value(subject_fwrfs[subj].rfs[1])[v, np.newaxis])[0], interpolation='None',
                   cmap=cm.coolwarm, origin='upper')  # vidxes[-100:]
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        # plt.title(r'$\rho=$%.2f' % subject_val_cc[subj][v], fontsize=16)
    # In[46]:
    filename = output_dir + 'S%02d/subj%02d_rf1_sample%s' % (subj, subj, saveext)
    fig2b.patch.set_alpha(0.)
    fig2b.savefig(filename, **savearg)
    plt.show()
    plt.close()