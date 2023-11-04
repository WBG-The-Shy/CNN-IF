from IPython.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import random
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

seed = 1
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

device = torch.device("cuda:0")  # cuda
trn_subjects = [1,2,3,5]
trn_str = '1,2,3,5'                     # select the training dataset and subject
model_name = 'GNet'                    # 'model_name_mpf_general'
ROI_name = "FFA_EBA_RSC_VWFA"
pretrained = False
quick_load = True
feature_filter = True
feature_filter_num = 512
feature_filter_size = 128
learning_rate = 1e-3
session = 16
batch_size = 50
num_epochs = 50
holdout_size = 1200
n_resample = 100
root_dir = os.getcwd() + '/'
net_dir = root_dir + "net/"
voxel_data_dir = "/code/voxel_data/" + ROI_name + '/' + trn_str + '/%dsession/voxel_data_subj:[%s]_upto:%d.h5py'%(session,trn_str,session)
filter_dir = "//code/" + 'feature filter/%s/%s/%d/' % (model_name, trn_str, feature_filter_num)
if pretrained:
    pre = "pretrained"
else:
    pre = "unpretrained"

saveext = ".png"
savearg = {'format': 'png', 'dpi': 120, 'facecolor': None}
timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())  # 'Aug-10-2020_1638' #

model_dir = '%s_%s' % (model_name, timestamp)

output_dir = "/Training_Result/untrained/%s/%s/%s/%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d/%s_%s/"%(
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

seed_everything(seed)

GNet = Encoder(trn_stim_mean, trunk_width=64).to(device)
if pretrained:
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url

    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', progress=True)
    ### Rename dictionary keys to match new breakdown
    pre_state_dict = {}
    pre_state_dict['conv1.0.weight'] = state_dict.pop('features.0.weight')
    pre_state_dict['conv1.0.bias'] = state_dict.pop('features.0.bias')
    pre_state_dict['conv2.0.weight'] = state_dict.pop('features.3.weight')
    pre_state_dict['conv2.0.bias'] = state_dict.pop('features.3.bias')

    GNet.pre.load_state_dict(pre_state_dict)

rec, fmaps, h = GNet(T.from_numpy(trn_stim_data[trn_subjects[0]][:1]).to(device))
for k, _fm in enumerate(fmaps):
    print(_fm.size())
subject_fwrfs = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv[s], pre_nl=_log_act_fn, \
                                        post_nl=_log_act_fn, dtype=np.float32).to(device) for s in trn_subjects}

for s, sp in subject_fwrfs.items():
    print("--------- subject %d ----------" % s)
    for p in sp.parameters():
        print("block size %-16s" % (list(p.size())))

param_count = 0
for w in GNet.parameters():
    param_count += np.prod(tuple(w.size()))
print('')
print(param_count, "shared params")
total_nv = 0
for s, sp in subject_fwrfs.items():
    for p in sp.parameters():
        param_count += np.prod(tuple(p.size()))
    total_nv += nnv[s]
print(param_count // total_nv, "approx params per voxels")


optimizer_net = optim.Adam([
    # {'params': shared_model.pre.parameters()},
    {'params': GNet.parameters()},
], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

subject_optimizer = {s: optim.Adam([
    {'params': sp.parameters()}
], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08) for s, sp in subject_fwrfs.items()}

subject_opts = {s: [optimizer_net, subject_optimizer[s]] for s in subject_optimizer.keys()}

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


batch_size = 12
num_epochs = 50
holdout_size = 1200
print(batch_size)
print(num_epochs)
print(holdout_size)
best_params, final_params, hold_cc_hist, hold_hist, trn_cc_hist, trn_hist, best_epoch, best_joint_cc_score = \
    learn_params_(output_dir, _training_fn, _holdout_fn, _pred_fn, GNet, subject_fwrfs, subject_opts,
                  trn_stim_data, trn_voxel_data,
                  num_epochs=num_epochs, batch_size=batch_size, holdout_size=holdout_size, masks=None, randomize=False)
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

In[29]:

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
# ---
# ## Final validation accuracy

val_voxel = {s: val_voxel_single_trial_data[s] for s in val_voxel_single_trial_data.keys()}
GNet.load_state_dict(best_params['enc'])
torch.save(GNet, output_dir + '%s_%s_%s_%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d.pth'%(
    model_name,ROI_name,pre,session,trn_str,num_epochs,batch_size,learning_rate,holdout_size))  # save the model
# save the weights
weights_dir = output_dir + '%s_%s_%s_%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d_weights'%(
    model_name,ROI_name,pre,session,trn_str,num_epochs,batch_size,learning_rate,holdout_size)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
a0 = best_params['fwrfs']
for i in trn_subjects:
    subj = a0[i]
    subjw = subj['w']
    subjb = subj['b']
    voxel_num = len(subjb)
    weights = subjw.cpu()
    weights = weights.numpy()
    bias = subjb.cpu()
    bias = bias.numpy()
    np.savetxt(weights_dir + "/%s_weights_subj%02d_%s.txt" % (ROI_name, i, voxel_num), weights, fmt="%s", delimiter=",")
    np.savetxt(weights_dir + "/%s_bias_subj%02d_%s.txt" % (ROI_name, i, voxel_num), bias, fmt="%s", delimiter=",")


GNet.eval()
for s, sd in subject_fwrfs.items():
    # sd.load_state_dict(best_params['fwrfs'][s])
    sd.eval()

batch_size = 16
subject_val_cc = validation_(_pred_fn, GNet, subject_fwrfs, val_stim_single_trial_data, val_voxel, batch_size)
joined_val_cc = np.concatenate(list(subject_val_cc.values()), axis=0)
volume_cc = {}
for k, s in enumerate(trn_subjects):
    subject_dir = output_dir + 'S%02d/' % s
    volume_cc[s] = view_data(brain_nii_shape[s], voxel_idx[s], subject_val_cc[s], save_to=subject_dir + "val_cc_%d"%s)
    np.savetxt(subject_dir + "GNet_data_subj%d_val.txt"%s, subject_val_cc[s], fmt="%s",
               delimiter=",")
# In[45]:

f = open(output_dir + 'val_acc.txt', "w+")
f.write("Run on subjects %s\n"%trn_str)


# print("best training joint score = %.3f\n" % best_joint_cc_score)
print("median joint val cc = %f" % np.median(joined_val_cc))  # the joint validation accuracy median
print("mean joint val cc = %f" % np.mean(joined_val_cc))  # the joint validation accuracy median
# f.write("best joint score = %.3f\n" % best_joint_cc_score) # the best joint cc score
f.write("median joint val cc = %f\n" % np.median(joined_val_cc)) # average among four subjects
f.write("mean joint val cc = %f\n" % np.mean(joined_val_cc)) # average among four subjects
for s, v in subject_val_cc.items():
    print("subject %s: median val cc = %f" % (s, np.median(v)))
    print("subject %s: mean val cc = %f" % (s, np.mean(v)))
    print("subject %s: max val cc = %f" % (s, np.max(v)))
    print("subject %s: min val cc = %f" % (s, np.min(v))) # the individual validation accuracy median
    f.write("\nsubject %s: median val cc = %f\n" % (s, np.median(v)))
    f.write("subject %s: mean val cc = %f\n" % (s, np.mean(v)))
    f.write("subject %s: max val cc = %f\n" % (s, max(v)))
    f.write("subject %s: min val cc = %f\n" % (s, min(v)))
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
# # Save final parameters

# In[32]:


torch.save({
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'holdout_size': holdout_size,
    'best_params': best_params,
    'final_params': final_params,
    'trn_loss_history': trn_hist,
    'hold_loss_history': hold_hist,
    'trn_cc_history': trn_cc_hist,
    'hold_cc_history': hold_cc_hist,
    'best_epoch': best_epoch,
    'best_joint_cc_score': best_joint_cc_score,
    'val_cc': subject_val_cc,
    'input_mean': trn_stim_mean,
    'input_std': trn_stim_std,
    'brain_nii_shape': brain_nii_shape,
    'voxel_index': voxel_idx,
    'voxel_roi': voxel_roi,
    'voxel_mask': voxel_mask,
    'voxel_ncsnr': voxel_ncsnr,
}, output_dir + 'model_params')

# In[33]:


f = open(output_dir + 'model_notes.txt', "w+")
f.write("Run on subjects %s\n"%trn_str)
f.close()


n_resample = 100
subject_resample_val_cc = {}

for s in val_voxel_single_trial_data.keys():
    print('sampling subject %d' % s)
    subject_resample_val_cc[s] = cc_resampling_with_replacement(_pred_fn, GNet, subject_fwrfs[s],
                                                                val_stim_single_trial_data[s],
                                                                val_voxel_single_trial_data[s], batch_size, n_resample)


f = open(output_dir + 'model_notes.txt', "a")
f.write("-- median_validation_accuracies --\n")
print("resample num = %d"%n_resample)
f.write("resample num = %d\n"%n_resample)
fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=0.2, bottom=0.60, right=.95, top=.95, wspace=0., hspace=0.)
val_score_samples = [np.median(np.concatenate([ccs[i] for s, ccs in subject_resample_val_cc.items()], axis=0)) \
                     for i in range(n_resample)]
print('score mean = %f, std.dev = %f' % (np.mean(val_score_samples), np.std(val_score_samples)))
f.write('score mean = %f, std.dev = %f\n' % (np.mean(val_score_samples), np.std(val_score_samples)))
plt.plot([0, ] * len(val_score_samples), val_score_samples, linestyle='None', marker='o', ms=10)
plt.errorbar(x=[0, ], y=[np.mean(val_score_samples), ], yerr=[np.std(val_score_samples), ], marker='o', \
             color='k', ms=10, elinewidth=4, capsize=16, capthick=4)
xticks = [i for i in range(0,9)]
plt.xticks(xticks)
for k, (s, ccs) in enumerate(subject_resample_val_cc.items()):  # this is subject-wise result
    mcc = [np.median(cc) for cc in ccs]
    print('Subject %s median cc mean = %f, std.dev = %f' % (s, np.mean(mcc), np.std(mcc)))
    f.write('Subject %s median cc mean = %f, std.dev = %f\n' % (s, np.mean(mcc), np.std(mcc)))
    plt.plot([s , ] * len(mcc), mcc, linestyle='None', marker='o', ms=10)
    plt.errorbar(x=[s , ], y=[np.mean(mcc), ], yerr=[np.std(mcc), ], marker='o', \
                 color='k', ms=10, elinewidth=4, capsize=16, capthick=4)
plt.ylabel('Median score')
plt.xlabel('validation subjects')
plt.savefig(output_dir + 'Median_score.jpg', dpi=800)
plt.show()
f.close()

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
        plt.title(r'$\rho=$%.2f' % subject_val_cc[subj][v], fontsize=16)
    # In[44]:
    filename = output_dir + 'S%02d/subj%02d_rf0_sample%s' % (subj, subj, saveext)
    fig2a.patch.set_alpha(0.)
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
        plt.title(r'$\rho=$%.2f' % subject_val_cc[subj][v], fontsize=16)
    # In[46]:
    filename = output_dir + 'S%02d/subj%02d_rf1_sample%s' % (subj, subj, saveext)
    fig2b.patch.set_alpha(0.)
    fig2b.savefig(filename, **savearg)
    plt.show()
    plt.close()