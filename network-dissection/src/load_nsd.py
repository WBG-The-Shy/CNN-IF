import sys
import os
import numpy as np
import h5py
from scipy.io import loadmat


def load_beta_file(filename, voxel_mask=None, zscore=True):
    from src.file_utility import load_mask_from_nii

    if '.hdf5' in filename: # .mat
        print('1')
        beta_data_set = h5py.File(filename, 'r')
        values = np.copy(beta_data_set['betas'])
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1), order='F').astype(np.float32) / 300.  #若是a.reshape(x, -1)则是将矩阵a变成行数为x，列数不规定的矩阵，具体列数按照总元素个数除行数，均分得到。
        else:
            beta = values.reshape((len(values), -1), order='F')[:,voxel_mask.flatten()].astype(np.float32) / 300.  #np.astype就是转换numpy数组的数据类型
        beta_data_set.close()
    elif ".nii" in filename:
        print('2')
        values = load_mask_from_nii(filename).transpose((3,0,1,2))
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1)).astype(np.float32) / 300.
        else:
            beta = values.reshape((len(values), -1))[:,voxel_mask.flatten()].astype(np.float32) / 300.
    else:
        print('3')
        print ("Unknown fdddile format")
        return None
    ###
    if zscore:
        mb = np.mean(beta, axis=0, keepdims=True)   # junzhi
        sb = np.std(beta, axis=0, keepdims=True)    # standard cha
        beta = np.nan_to_num((beta - mb) / (sb + 1e-6))
        print ("<beta> = %.3f, <sigma> = %.3f" % (np.mean(mb), np.mean(sb)))
    return beta
     


def load_betas(folder_name, zscore=False, voxel_mask=None, up_to=0, load_ext='.hdf5'):
    '''load beta value in the structure of the NSD experiemnt'''
    from src.file_utility import list_files #read file  import files
    matfiles, betas = [], []
    k = 0
    for filename in list_files(folder_name):
        filename_no_path = filename.split('/')[-1] # filename = url.split('/')[-1]  #以‘/ ’为分割f符，保留最后一段
        if 'betas' in filename_no_path and load_ext in filename_no_path:
            k += 1
            if up_to>0 and k>up_to:
                break
            print (filename) 
            matfiles += [filename,]  
            betas += [ load_beta_file(filename, voxel_mask=voxel_mask, zscore=zscore), ]         # zscore
    return np.concatenate(betas, axis=0), matfiles
    
    
def image_feature_fn(image):
    '''take uint8 image and return floating point (0,1), either color or bw'''
    return image.astype(np.float32) / 255   #change type into float32

def image_uncolorize_fn(image):
    data = image.astype(np.float32) / 255
    return (0.2126*data[:,0:1]+ 0.7152*data[:,1:2]+ 0.0722*data[:,2:3])
    
    
    
def ordering_split(voxel, ordering, combine_trial=False):
    data_size, nv = voxel.shape 
    print ("Total number of voxels = %d" % nv)
    ordering_data = ordering[:data_size]
    shared_mask = ordering_data<1000  # the first 1000 indices are the shared indices
    
    if combine_trial:        
        idx, idx_count = np.unique(ordering_data, return_counts=True)
        idx_list = [ordering_data==i for i in idx]
        voxel_avg_data = np.zeros(shape=(len(idx), nv), dtype=np.float32)
        for i,m in enumerate(idx_list):
            voxel_avg_data[i] = np.mean(voxel[m], axis=0)
        shared_mask_mt = idx<1000

        val_voxel_data = voxel_avg_data[shared_mask_mt] 
        val_stim_ordering = idx[shared_mask_mt]   

        trn_voxel_data = voxel_avg_data[~shared_mask_mt]
        trn_stim_ordering = idx[~shared_mask_mt]              
        
    else:
        val_voxel_data = voxel[shared_mask]    
        val_stim_ordering  = ordering_data[shared_mask]

        trn_voxel_data = voxel[~shared_mask]
        trn_stim_ordering  = ordering_data[~shared_mask]
        
    return trn_stim_ordering, trn_voxel_data, val_stim_ordering, val_voxel_data

# the content of ordering is the sequence of 10000 images
# stim 10000*3*227*227   voxel 3000*238292
def data_split(stim, voxel, ordering, imagewise=True):
    data_size, nv = voxel.shape 
    print ("Total number of voxels = %d" % nv)
    ordering_data = ordering[:data_size]   # 从下标i到下标j，截取序列s中的元素。 ordering.size = 30000
    #the first 3000 order data
    shared_mask = ordering_data<1000  # the first 1000 indices are the shared indices if ordering_data<1000 then shared_mask = True

    #single trial
    val_voxel_st = voxel[shared_mask]    # the data of shared 1000 images voxel data
    val_stim_st  = stim[ordering_data[shared_mask]] #the data of shared 1000 images stim data
    #return_count为True时：会构建一个递增的唯一值的新列表，并返回新列表values 中的值在旧列表中的个数 counts
    idx, idx_count = np.unique(ordering_data, return_counts=True)  # delete the same num
    idx_list = [ordering_data==i for i in idx]
    voxel_avg_data = np.zeros(shape=(len(idx), nv), dtype=np.float32)
    for i,m in enumerate(idx_list):
        voxel_avg_data[i] = np.mean(voxel[m], axis=0)  #axis=0，计算每一列的均值
    shared_mask_mt = idx<1000

    #mutl trial
    val_voxel_mt = voxel_avg_data[shared_mask_mt]  
    val_stim_mt  = stim[idx][shared_mask_mt]        
    
    if imagewise:
        trn_voxel = voxel_avg_data[~shared_mask_mt]
        trn_stim  = stim[idx][~shared_mask_mt] 
        return trn_stim, trn_voxel, val_stim_st, val_voxel_st, val_stim_mt, val_voxel_mt
    else:
        trn_voxel = voxel[~shared_mask]
        trn_stim = stim[ordering_data[~shared_mask]]
        return trn_stim, trn_voxel, val_stim_st, val_voxel_st, val_stim_mt, val_voxel_mt