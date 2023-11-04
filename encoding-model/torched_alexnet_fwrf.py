from IPython.display import display, HTML
import random
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
device = torch.device("cuda:0")  # cuda

torch.manual_seed(time.time())
torch.backends.cudnn.enabled = True
torch.manual_seed(time.time())       #system time       time.clock   is  cpu time
torch.backends.cudnn.enabled=True #
#torch.backends.cudnn.bencharmk = True
sns.axes_style()
sns.set_style("whitegrid", {"axes.facecolor": '.95'})
sns.set_context("notebook",
                rc={'axes.labelsize': 18.0, 'axes.titlesize': 24.0, 'legend.fontsize': 18.0, 'xtick.labelsize': 18.0,
                    'ytick.labelsize': 18.0})
sns.set_palette("deep")

device = torch.device("cuda:0")  # cuda
subject = 1
trn_subjects = [1]
trn_str = '1'                     # select the training dataset and subject
model_name = 'AlexNet'                    # 'model_name_mpf_general'
ROI_name = "V1_V2_V3_V4_FFA_EBA_RSC_VWFA"
pretrained = True
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
voxel_data_dir = "" + ROI_name + '/' + trn_str + '/%dsession/voxel_data_subj:[%s]_upto:%d.h5py'%(session,trn_str,session)
filter_dir = "" + 'feature filter/%s/%s/%d/' % (model_name, trn_str, feature_filter_num)
if pretrained:
    pre = "pretrained"
else:
    pre = "unpretrained"

saveext = ".png"
savearg = {'format': 'png', 'dpi': 120, 'facecolor': None}
timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())  # 'Aug-10-2020_1638' #

model_dir = '%s_%s' % (model_name, timestamp)

output_dir = "/%s/%s/%s/%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d/%s_%s/"%(
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
nsd_root = "/home/mufan/VDisk1/Mufan/nsd/"
stim_root = nsd_root + "nsddata_stimuli/stimuli/nsd/"
beta_root = nsd_root + "nsddata_betas/ppdata/"
mask_root = nsd_root + "nsddata/ppdata/"

exp_design_file = nsd_root + "nsddata/experiments/nsd/nsd_expdesign.mat"

exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1  # zero-indexed ordering of indices (matlab-like to python-like))
print(ordering.size)    #hang discrease dim
image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
image_data = np.copy(image_data_set['stimuli'])   #deep copy   different address    stimuli data size = 10000*3*227*227
# print(image_data)   #nparray
image_data_set.close()
print ('block size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:',\
    np.min(image_data[0]), np.max(image_data[0]))
print ('device#:', torch.cuda.current_device())

from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi

group_names = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA','RSC','VWFA']
group = [[1], [2], [3], [4], [5], [6], [7], [8]]
brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

print('--------  subject %d  -------' % s)
# load mask and roi
visual_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % subject)
ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % subject)
eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % subject)
rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % subject)
vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % subject)
a = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % subject)
b = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % subject)
# turn nii file into ndarry for next process
a = a.flatten()
b = b.flatten()
ffa_full = ffa_full.flatten()
eba_full = eba_full.flatten()
rsc_full = rsc_full.flatten()
vwfa_full = vwfa_full.flatten()
visual_full = visual_full.flatten()
voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % subject)
voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % subject)
voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (subject))
general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (subject))
ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz" % subject)
count_ffa = 0
count_eba = 0
count_rsc = 0
count_vwfa = 0
count_visual = 0
count_mix = 0
counta = 0
# count the number of voxels and make a jointed roi
for i in range(len(ffa_full)):
            a[i] = 0
            b[i] = 0

            if ffa_full[i] < 1:
                ffa_full[i] = 0
            if eba_full[i] < 1:
                eba_full[i] = 0
            if rsc_full[i] < 1:
                rsc_full[i] = 0
            if vwfa_full[i] < 1:
                vwfa_full[i] = 0
            if visual_full[i] < 1:
                visual_full[i] = 0

            if ffa_full[i] > 0:
                count_ffa = count_ffa + 1
                ffa_full[i] = 5
            if eba_full[i] > 0:
                count_eba = count_eba + 1
                eba_full[i] = 6
            if rsc_full[i] > 0:
                count_rsc = count_rsc + 1
                rsc_full[i] = 7
            if vwfa_full[i] > 0:
                count_vwfa = count_vwfa + 1
                vwfa_full[i] = 8
            if visual_full[i] > 0:
                count_visual = count_visual + 1

            if ffa_full[i] > 0 and visual_full[i] > 0:
                count_mix = count_mix + 1
            if visual_full[i] > 0:
                if visual_full[i] == 1 or visual_full[i] == 2:
                    b[i] = 1
                if visual_full[i] == 3 or visual_full[i] == 4:
                    b[i] = 2
                if visual_full[i] == 5 or visual_full[i] == 6:
                    b[i] = 3
                if visual_full[i] == 7:
                    b[i] = 4
            if ffa_full[i] > 0:
                b[i] = ffa_full[i]  # if voxel in visual roi and ffa roi think it in ffa roi
            if eba_full[i] > 0:
                b[i] = eba_full[i]
            if rsc_full[i] > 0:
                b[i] = rsc_full[i]
            if vwfa_full[i] > 0:
                b[i] = vwfa_full[i]
            if visual_full[i] > 0 or ffa_full[i] > 0 or eba_full[i]>0 or rsc_full[i]>0 or vwfa_full[i]>0:
                a[i] = 1
                counta = counta + 1
print("ffa voxels = %d " % count_ffa)
print("eba voxels = %d " % count_eba)
print("rsc voxels = %d " % count_rsc)
print("vwfa voxels = %d " % count_vwfa)
print("visual voxels = %d " % count_visual)
print("overlap voxels = %d " % count_mix)
print("joint roi voxels = %d " % counta)
brain_nii_shape = voxel_roi_full.shape
print(brain_nii_shape)
###
voxel_roi_mask_full = (a > 0).astype(bool)
voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois

voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
 ###
voxel_mask = a.flatten()
voxel_mask = voxel_mask.astype(bool)
voxel_idx = np.arange(len(voxel_mask))[voxel_mask]
voxel_roi = b[voxel_mask]
voxel_ncsnr = ncsnr_full.flatten()[voxel_mask]

print('full mask length = %d' % len(voxel_mask))
print('selection length = %d' % np.sum(voxel_mask))

for roi_mask, roi_name in iterate_roi(group, voxel_roi, roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))

slice_idx = 35
plt.figure(figsize=(12, 4 * len(trn_subjects)))
subject_dir = output_dir + 'S%02d/' % subject
if not os.path.exists(subject_dir):
    os.makedirs(subject_dir)

volume_brain_mask = view_data(brain_nii_shape, voxel_idx, np.ones_like(voxel_idx),
                                  save_to=subject_dir + "subj%02d_mask" % subject)
volume_brain_roi = view_data(brain_nii_shape, voxel_idx, voxel_roi,
                                 save_to=subject_dir + "subj%02d_roi" % subject)
volume_ncsnr      = view_data(brain_nii_shape, voxel_idx, voxel_ncsnr, save_to=subject_dir+"subj%02d_ncsnr"%subject)
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

from src.load_nsd import load_betas
from src.load_nsd import image_feature_fn, data_split

if quick_load:
    voxel_data_set = h5py.File(voxel_data_dir, 'r')
    # voxel_data_set = h5py.File(root_dir+'voxel_data_general.h5py', 'r')
    voxel_data_dict = embed_dict({k: np.copy(d) for k, d in voxel_data_set.items()})
    voxel_data_set.close()
    voxel_data = voxel_data_dict['voxel_data']

    data_size, nv = voxel_data.shape
    trn_stim_data, trn_voxel_data, \
        val_stim_single_trial_data, val_voxel_single_trial_data, \
        val_stim_multi_trial_data, val_voxel_multi_trial_data = \
        data_split(image_feature_fn(image_data), voxel_data, ordering, imagewise=False)

    del voxel_data
else:
    print('--------  subject %d  -------' % subject)
    beta_subj = beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/" % subject
    voxel_data, _ = load_betas(folder_name=beta_subj, zscore=True, voxel_mask=voxel_mask, up_to=session)
    print('----------------------------')
    print(voxel_data.shape)
    voxel_dir = "/code/voxel_data/%s/%s/%dsession/" % (ROI_name, trn_str, session)
    if not os.path.exists(voxel_dir):
        os.makedirs(voxel_dir)
    # up_to = 1
    # root_dir + 'voxel_data/%s/%s/%dsession/voxel_data_subj:%s_upto_:%d'%(ROI_name,trn_str,session, trn_str,session)
    save_stuff(voxel_dir + '/voxel_data_subj:[%s]_upto:%d' % (trn_str, session),
               flatten_dict({
                   'voxel_mask': voxel_mask,
                   'voxel_roi': voxel_roi,
                   'voxel_idx': voxel_idx,
                   'voxel_ncsnr': voxel_ncsnr,
                   'voxel_data': voxel_data
               }))
    data_size, nv = voxel_data.shape
    trn_stim_data, trn_voxel_data, \
        val_stim_single_trial_data, val_voxel_single_trial_data, \
        val_stim_multi_trial_data, val_voxel_multi_trial_data = \
        data_split(image_feature_fn(image_data), voxel_data, ordering, imagewise=False)

    del voxel_data

from src.torch_fwrf import get_value, set_value
from models.alexnet import Alexnet_fmaps
from models.vgg import VGG16_fmaps
import torch.nn as nn
_fmaps_fn = Alexnet_fmaps().to(device)    #create the model
# _fmaps_fn1 = VGG16_fmaps().to(device)
_x = torch.tensor(trn_stim_data[:1]).to(device)
_fmaps = _fmaps_fn(_x)   # get the feature map of each layer   with the input of the 100th stim images
for k,_fm in enumerate(_fmaps):
    print (_fm.size())



from src.torch_feature_space import filter_dnn_feature_maps
_fmaps_fn, lmask, fmask, tuning_masks = filter_dnn_feature_maps(image_data, _fmaps_fn, batch_size=feature_filter_size, fmap_max=feature_filter_num)

_x = torch.tensor(image_data[:1]).to(device) # the input variable.
_fmaps = _fmaps_fn(_x)
for k,_fm in enumerate(_fmaps):     #the new _fmaps_fn
    print (_fm.size())              #1:64+192    2:256+256+256   3:512+512+512

from PIL import Image
from matplotlib.patches import Patch
# im_frame = Image.open(root_dir + 'model_diagram_paper.png')
im_frame = Image.open(root_dir + 'model_diagram_paper.png')  # 1268*1252
cmap = np.array(im_frame.getdata()).reshape((1268,1252,4))[:10]   # the fisrt 10th   1268*1252*4
print(cmap.shape)


# In[20]:


layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
fig=plt.figure(figsize=(8, 4))
plt.subplots_adjust(left=0.01, bottom=0.35, right=.65, top=.95, wspace=0.16, hspace=0.)
legend_elements = [Patch(facecolor=tuple(cmap[0,int(float(k)*255./(len(tuning_masks)-1))]/255), edgecolor=tuple(cmap[0,int(float(k)*255./(len(tuning_masks)-1))]/255), label='%d: %s'%(k, layer_names[k])) for k,tm in enumerate(tuning_masks)]
for k,tm in enumerate(tuning_masks):
    _=plt.plot(tm, marker='|', linestyle='None', color=tuple(cmap[0,int(float(k)*255./(len(tuning_masks)-1))]/255))
#_=plt.title('Feature index layer correspondence')
_=plt.xlabel('Feature index')
_=plt.ylim([.95, 1.05])
_=plt.yticks([])
_=plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.1))
plt.savefig(output_dir + 'feature_index.jpg', dpi=800)
plt.show()

filename = output_dir + "layer_feature_map%s" % (saveext)
fig.savefig(filename, **savearg)
plt.close()

# # fwRF training procedure
# The model is
# $$ r(t) = b + W * [f(\int_\mathrm{space}\phi(x,y,t) * g(x,y) dxdy) - m] / \sigma $$
# where
# $g(x,y)$ is a gaussian pooling field shared by all feature maps
#
# $\phi(x,y,t)$ are the feature maps corresponding to stimuli $t$
#
# $W, b$ are the feature weights and bias of the linearized model for each voxels
#
# $f(\cdot)$ is an optional nonlinearity
#
# $m,\sigma$ are normalization coefficient to facilitate regularization

from src.rf_grid    import linspace, logspace, model_space, model_space_pyramid
from src.torch_fwrf import learn_params_ridge_regression, get_predictions

aperture = np.float32(1)  # aperture 孔，穴；（照相机，望远镜等的）光圈，孔径；缝隙
nx = ny = 11
smin, smax = np.float32(0.05), np.float32(0.4)
ns = 8

# sharedModel specification is a list of 3 ranges and 3 callable functor. The reason for this is for a future implementation of dynamic mesh refinement.
#model_specs 说明书 = [[(0., aperture*1.1), (0., aperture*1.1), (smin, smax)], [linspace(nx), linspace(ny), logspace(ns)]]
#models = model_space(model_specs)
models = model_space_pyramid(logspace(ns)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)
print ('candidate count = ', len(models))

lambdas = np.logspace(3.,7.,9, dtype=np.float32)
#_log_act_func = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(torch.abs(_x))

from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx

lx, vx = aperture, aperture * 1.5
cNorm  = colors.Normalize(vmin=.0, vmax=.4)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet') )

fig=plt.figure(figsize=(8, 8))
plt.title('Grid search candidates (%d candidates)' % len(models))
plt.plot(models[:,0], models[:,1], '.k', linestyle='None')
ax = plt.gca()
for s in np.unique(models[:,2]):
    ax.add_artist(Ellipse(xy=(0,0), width=2*s, height=2*s, angle=0,
        color=scalarMap.to_rgba(s), lw=1, fill=False))
_=ax.set_xlim(-vx/2, vx/2)
_=ax.set_ylim(-vx/2, vx/2)
_=ax.set_aspect('equal')
plt.plot([-lx/2,lx/2,lx/2,-lx/2,-lx/2], [lx/2,lx/2,-lx/2,-lx/2, lx/2], 'r', lw=2)
plt.savefig(output_dir + 'grid_search_candidates.jpg', dpi=800)
plt.show()

filename = output_dir + "rf_grid%s" % (saveext)
fig.savefig(filename, **savearg)
plt.close()

from src.torch_fwrf import  learn_params_ridge_regression, get_predictions, Torch_fwRF_voxel_block

sample_batch_size = 2000
voxel_batch_size = 1000 # throughput 总吞吐量

best_losses, best_lambdas, best_params = learn_params_ridge_regression(
    trn_stim_data, trn_voxel_data, _fmaps_fn, models, lambdas, \
    aperture=aperture, _nonlinearity=None, zscore=True, sample_batch_size=sample_batch_size, \
    voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=False, add_bias=True)

print ([p.shape if p is not None else None for p in best_params])

weights_dir = output_dir + '%s_%s_%s_%dsession_subj%s_%depoch_%dbs_lr=%.4f_hold=%d_weights'%(
    model_name,ROI_name,pre,session,trn_str,num_epochs,batch_size,learning_rate,holdout_size)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
weights = best_params[1]
bias = best_params[2]
voxel_num = len(bias)
np.savetxt(weights_dir + "/%s_weights_subj%02d_%s.txt" % (ROI_name, subject, voxel_num), weights, fmt="%s", delimiter=",")
np.savetxt(weights_dir + "/%s_bias_subj%02d_%s.txt" % (ROI_name, subject, voxel_num), bias, fmt="%s", delimiter=",")

# In[30]:

torch.save({
        'lmask': lmask,
        'fmask': fmask,
        'tuning_masks': tuning_masks,
        'aperture': aperture,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': np.array(brain_nii_shape),
        'val_size': len(val_voxel_single_trial_data),
        'trn_size': len(trn_voxel_data),
        'ordering': ordering,
        'voxel_index': voxel_idx,
        'voxel_roi': voxel_roi,
        'params': best_params,
        'lambdas': lambdas,
        'best_lambdas': best_lambdas,
    }, output_dir+'model_params')

torch.cuda.empty_cache()
sample_batch_size = 1000
voxel_batch_size = 500 # throughput 总吞吐量

from src.torch_fwrf import  learn_params_ridge_regression, get_predictions, Torch_fwRF_voxel_block

param_batch = [p[:voxel_batch_size] if p is not None else None for p in best_params]
_fwrf_fn = Torch_fwRF_voxel_block(_fmaps_fn, param_batch, _nonlinearity=None, input_shape=image_data.shape, aperture=1.0) # make a fwrf model

voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn, _fwrf_fn, best_params, sample_batch_size=sample_batch_size)

#val_voxel_pred = voxel_pred[ordering[:data_size][shared_mask]]
val_cc  = np.zeros(shape=(nv), dtype=fpX)
for v in tqdm(range(nv)):
    val_cc[v] = np.corrcoef(val_voxel_single_trial_data[:,v], voxel_pred[:,v])[0,1]
val_cc = np.nan_to_num(val_cc)


# In[33]:


import matplotlib.pyplot as plt
import pandas as pd
f = open(output_dir + 'val_acc.txt', "w+")
f.write("Run on subjects %s\n"%subject)
print("median  val cc = %.3f" % np.median(val_cc))  # the joint validation accuracy median
f.write("median  val cc = %.3f\n" % np.median(val_cc)) # average among four subjects
print("mean  val cc = %.3f" % np.mean(val_cc))  # the joint validation accuracy median
f.write("mean  val cc = %.3f\n" % np.mean(val_cc)) # average among four subjects
print("max  val cc = %.3f" % np.max(val_cc))  # the joint validation accuracy median
f.write("max  val cc = %.3f\n" % np.max(val_cc)) # average among four subjects
print("min  val cc = %.3f" % np.min(val_cc))  # the joint validation accuracy median
f.write("min  val cc = %.3f\n" % np.min(val_cc)) # average among four subjects
# data = pd.read_csv("/nd_disk3/guoyuan/Mufan/result/3000/val_cc.csv")
fig=plt.figure(figsize=(6,4))
plt.subplots_adjust(left=0., bottom=0.2, right=1., top=1., wspace=0., hspace=0.)
_=plt.hist(val_cc, bins=100, density=True, range=(-.5, 1.))
# _=plt.vlines(x=[0], ymin=1e-4, ymax=25, color='r')
_=plt.yscale('log')
_=plt.xlim([-.5, 1.0])
_=plt.xlabel(r'$\rho$')
_=plt.ylabel('Relative frequency')
plt.savefig(output_dir + 'relative_frequency.jpg', dpi=800)
plt.show()

plt.hist(val_cc,bins=1000,color='pink',edgecolor='b')
plt.savefig(output_dir + 'frequency.jpg', dpi=800)
plt.show()

best_models = best_params[0]
best_ecc  = np.sqrt(np.square(best_models[:,0]) + np.square(best_models[:,1]))
best_ang  = np.arctan2(best_models[:,1], best_models[:,0])
best_size = best_models[:,2]

volume_loss = view_data(brain_nii_shape, voxel_idx, best_losses,save_to=output_dir+"best_losses")
volume_cc   = view_data(brain_nii_shape, voxel_idx, val_cc, save_to=output_dir+"val_cc")
volume_ecc  = view_data(brain_nii_shape, voxel_idx, best_ecc, save_to=output_dir+"rf_ecc")
volume_ang  = view_data(brain_nii_shape, voxel_idx, best_ang, save_to=output_dir+"rf_ang")
volume_size = view_data(brain_nii_shape, voxel_idx, best_size, save_to=output_dir+"rf_size")

slice_idx = 40
fig = plt.figure(figsize=(30,6))
plt.subplot(1,5,1)
plt.title('Loss')
plt.imshow(volume_loss[:,:,slice_idx], cmap='gray', interpolation='None')
plt.colorbar()
plt.subplot(1,5,2)
plt.title('Validaton accuracy')
plt.imshow(volume_cc[:,:,slice_idx], cmap='hot', interpolation='None')
plt.colorbar()
plt.subplot(1,5,3)
plt.title('RF Eccentricity')
plt.imshow(volume_ecc[:,:,slice_idx], cmap='jet', interpolation='None')
plt.colorbar()
plt.subplot(1,5,4)
plt.title('RF Angle')
plt.imshow(volume_ang[:,:,slice_idx], cmap='hsv', interpolation='None')
plt.colorbar()
plt.subplot(1,5,5)
plt.title('RF Size')
plt.imshow(volume_size[:,:,slice_idx], cmap='jet', interpolation='None')
plt.colorbar()
plt.savefig(output_dir + 'show_NIFTI_result.jpg', dpi=800)
plt.show()

torch.save({
        'lmask': lmask,
        'fmask': fmask,
        'tuning_masks': tuning_masks,
        'aperture': aperture,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': np.array(brain_nii_shape),
        'val_size': len(val_voxel_single_trial_data),
        'trn_size': len(trn_voxel_data),
        'ordering': ordering,
        'voxel_index': voxel_idx,
        'voxel_roi': voxel_roi,
        'params': best_params,
        'lambdas': lambdas,
        'best_lambdas': best_lambdas,
        'val_cc': val_cc,
    }, output_dir+'model_params')
