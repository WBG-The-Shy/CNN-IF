import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math

import torch
import torch.nn as nn

def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)  #tranform array into tensor

def iterate_range(start, length, batchsize):     #iterater
    batch_count = int(length // batchsize )     # compute the iterate num    10000/100=100
    residual = int(length % batchsize)   #残留的；（数量）剩余的；
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
        # yield almostly = return but the next process will start from yeild
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        

class Torch_filter_fmaps(nn.Module):
    def __init__(self, _fmaps, lmask, fmask):
        super(Torch_filter_fmaps, self).__init__()
        device = next(_fmaps.parameters()).device

        self.fmaps = _fmaps
        self.lmask = lmask
        self.fmask = [nn.Parameter(torch.from_numpy(fm).to(device), requires_grad=False) for fm in fmask]
        for k,fm in enumerate(self.fmask):
             self.register_parameter('fm%d'%k, fm)

    def forward(self, _x):
        _fmaps = self.fmaps(_x)
        return [torch.index_select(torch.cat([_fmaps[l] for l in lm], axis=1), dim=1, index=fm) for lm,fm in zip(self.lmask, self.fmask)]



def get_tuning_masks(layer_rlist, fmaps_count):
    tuning_masks = []
    for rl in layer_rlist:
        tm = np.zeros(shape=(fmaps_count,), dtype=bool)
        tm[rl] = True
        tuning_masks += [tm,]
    return tuning_masks

def filter_dnn_feature_maps(data, _fmaps_fn, batch_size, fmap_max=1024, concatenate=False, trn_size=None):
    '''Runs over the image set and keep the fmap_max features with the most variance withing each layer of the network.
    Return an updated torch function and a list of binary mask that match the new feature space to identify the layer provenance of the feature'''
    device = next(_fmaps_fn.parameters()).device   #整句代码就是指定 使用和参数相同的设备
    #运行图像集，并在网络的每一层中保持fmap_max特性的最大差异。返回一个更新的torch function和一个匹配新特征空间的二进制掩码列表，以识别特征的层来源" '
    size = trn_size if trn_size is not None else len(data)
    # lambda x:  the function of x                                            transform x from numpy array into tensor
    fmaps_fn = lambda x: [np.copy(_fm.data.cpu().numpy()) for _fm in _fmaps_fn(_to_torch(x, device=device))]
    fmaps = fmaps_fn(data[:batch_size])
    #data_size=10000  batch_size=100       put batch size into _fmaps_fn  and get its feature maps   fmaps:list 8 layers
    run_avg = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]  #get an array size = fmaps.shape[1]
    # 64,192,384,256,256,4096,4096,1000      avg()用于对指定的列或表达式求平均值。
    run_sqr = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]   #一个非负实数的算术平方根
    for rr,rl in tqdm(iterate_range(0, size, batch_size)):
        #tqdm是 Python 进度条库，可以在 Python长循环中添加一个进度提示信息.
        fb = fmaps_fn(data[rr])  # get each batch size of image data
        for k,f in enumerate(fb):
            #k is num   f is each feature map
            #如果我们要用平均数来减少特征图的数量，我们只需要平均数
            #np.mean    axis=（i，j），即沿着数组第i和第j两个下标的变化方向`进行操作。 get the avg of 27*27 data
            if f.shape[1]>fmap_max: # only need the average if we're going to use them to reduce the number of feature maps
                run_avg[k] += np.sum(np.mean(f.astype(np.float64), axis=(2,3)), axis=0)    #lie sum    the avg sum of batch_size
                run_sqr[k] += np.sum(np.mean(np.square(f.astype(np.float64)), axis=(2,3)), axis=0)   #lie sum
                #函数返回一个新数组，该数组的元素值为源数组元素的平方。 源阵列保持不变。
    for k in range(len(fb)):   # average the run_avg run_sqr
        run_avg[k] /= size
        run_sqr[k] /= size
    fmask = [np.zeros(shape=(fm.shape[1]), dtype=bool) for fm in fmaps]
    fmap_var = [np.zeros(shape=(fm.shape[1]), dtype=np.float32) for fm in fmaps]
    for k,fm in enumerate(fmaps):
        if fm.shape[1]>fmap_max:
            #select the feature map with the most variance to the dataset  选择与数据集方差最大的特征图
            fmap_var[k] = (run_sqr[k] - np.square(run_avg[k])).astype(np.float32)
            #np.argsort() 将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出。   [-x:] 表示 最后 x 个元素构成的切片
            most_var = fmap_var[k].argsort()[-fmap_max:] #the feature indices with the top-fmap_max variance top-fmap方差最大的特征指标
            # print(np.sort(most_var))
            fmaps[k] = fm[:,np.sort(most_var),:,:]
            fmask[k][most_var] = True
        else:
            fmask[k][:] = True
        print ("layer: %s, shape=%s" % (k, (fmaps[k].shape)))
        sys.stdout.flush()

    # ORIGINAL PARTITIONING OF LAYERS         原始层划分
    fmaps_sizes = [fm.shape for fm in fmaps]
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    layer_rlist = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    # concatenate fmaps of identical dimension to speed up rf application  连接相同尺寸的fmap以加快rf receiptive field应用
    clmask, cfmask, cfmaps = [],[],[]
    print ("")
    sys.stdout.flush()
    # I would need to make sure about the order and contiguousness of the fmaps to preserve the inital order.
    # It isn't done right now but since the original feature maps are monotonically decreasing in resultion in
    # the examples I treated, the previous issue doesn't arise.
    # 我需要确保fmap的顺序和连续性，以保持初始顺序。
    # 现在还没有完成，但由于原始的特征映射是单调减少的结果
    # 我处理的例子，之前的问题没有出现。
    if concatenate: # 连接   np.prod 返回给定轴上的数组元素的乘积。
        #[::-1]取从后向前（相反）的元素
        #np.unique 一维数组或者列表，np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
        for k,us in enumerate(np.unique([np.prod(fs[2:4]) for fs in fmaps_sizes])[::-1]): ## they appear sorted from small to large, so I reversed the order 它们从小到大排序，所以我把顺序颠倒了
            mask = np.array([np.prod(fs[2:4])==us for fs in fmaps_sizes]) # mask over layers that have that spatial size
            lmask = np.arange(len(fmaps_sizes))[mask] # list of index for layers that have that size
            bfmask = np.concatenate([fmask[l] for l in lmask], axis=0)
            clmask += [lmask,]
            cfmask += [np.arange(len(bfmask))[bfmask],]
            cfmaps += [np.concatenate([fmaps[l] for l in lmask], axis=1),]
            print ("fmaps: %s, shape=%s" % (k, (cfmaps[-1].shape)))
            sys.stdout.flush()
        fmaps_sizes = [fm.shape for fm in cfmaps]
    else:
        for k,fm in enumerate(fmask):
            clmask += [np.array([k]),]
            cfmask += [np.arange(len(fm))[fm],]
    ###
    tuning_masks = get_tuning_masks(layer_rlist, fmaps_count)
    assert np.sum(sum(tuning_masks))==fmaps_count, "%d != %d" % (np.sum(sum(tuning_masks)), fmaps_count)
    #layer_rlist, fmaps_sizes, fmaps_count, clmask, cfmask #, scaling
    return Torch_filter_fmaps(_fmaps_fn, clmask, cfmask), clmask, cfmask, tuning_masks


'''
def filter_dnn_feature_maps(data, _fmaps_fn, batch_size, fmap_max=1024, concatenate=True, trn_size=None):
    Runs over the image set and keep the fmap_max features with the most variance withing each layer of the network.
    Return an updated torch function and a list of binary mask that match the new feature space to identify the layer provenance of the feature
    # device = next(_fmaps_fn.parameters()).device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.cuda.set_device("cuda:0")
    size = trn_size if trn_size is not None else len(data)
#    _fmaps = _fmaps_fn(_to_torch(data[:batch_size], device=device))
    fmaps_fn = lambda x: [np.copy(_fm.data.cuda()) for _fm in _fmaps_fn(_to_torch(x, device=device))]  #_fm.data.cpu().numpy()
    # fmaps = fmaps_fn(data[:batch_size],device=device)
    fmaps = fmaps_fn(_to_torch(data[:batch_size], device=device))
    # print(fmaps_fn.is_cuda, fmaps.is_cuda)
    run_avg = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]
    run_sqr = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]
    for rr,rl in tqdm(iterate_range(0, size, batch_size)):
        fb = fmaps_fn(data[rr])
        for k,f in enumerate(fb):
            if f.shape[1]>fmap_max: # only need the average if we're going to use them to reduce the number of feature maps
                run_avg[k] += np.sum(np.mean(f.astype(np.float64), axis=(2,3)), axis=0)
                run_sqr[k] += np.sum(np.mean(np.square(f.astype(np.float64)), axis=(2,3)), axis=0)
    for k in range(len(fb)):
        run_avg[k] /= size
        run_sqr[k] /= size
    ###
    fmask = [np.zeros(shape=(fm.shape[1]), dtype=bool) for fm in fmaps]
    fmap_var = [np.zeros(shape=(fm.shape[1]), dtype=np.float32) for fm in fmaps]
    for k,fm in enumerate(fmaps):  
        if fm.shape[1]>fmap_max:
            #select the feature map with the most variance to the dataset
            fmap_var[k] = (run_sqr[k] - np.square(run_avg[k])).astype(np.float32)
            most_var = fmap_var[k].argsort()[-fmap_max:] #the feature indices with the top-fmap_max variance
            fmaps[k] = fm[:,np.sort(most_var),:,:]
            fmask[k][most_var] = True
        else:
            fmask[k][:] = True
        print ("layer: %s, shape=%s" % (k, (fmaps[k].shape)))    
        sys.stdout.flush()

    # ORIGINAL PARTITIONING OF LAYERS
    fmaps_sizes = [fm.shape for fm in fmaps]
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])   
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    layer_rlist = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    # concatenate fmaps of identical dimension to speed up rf application
    clmask, cfmask, cfmaps = [],[],[]
    print ("")
    sys.stdout.flush()
    # I would need to make sure about the order and contiguousness of the fmaps to preserve the inital order.
    # It isn't done right now but since the original feature maps are monotonically decreasing in resultion in
    # the examples I treated, the previous issue doesn't arise.
    if concatenate:
        for k,us in enumerate(np.unique([np.prod(fs[2:4]) for fs in fmaps_sizes])[::-1]): ## they appear sorted from small to large, so I reversed the order
            mask = np.array([np.prod(fs[2:4])==us for fs in fmaps_sizes]) # mask over layers that have that spatial size
            lmask = np.arange(len(fmaps_sizes))[mask] # list of index for layers that have that size
            bfmask = np.concatenate([fmask[l] for l in lmask], axis=0)
            clmask += [lmask,]
            cfmask += [np.arange(len(bfmask))[bfmask],]
            cfmaps += [np.concatenate([fmaps[l] for l in lmask], axis=1),]
            print ("fmaps: %s, shape=%s" % (k, (cfmaps[-1].shape)))
            sys.stdout.flush()
        fmaps_sizes = [fm.shape for fm in cfmaps]
    else:
        for k,fm in enumerate(fmask):
            clmask += [np.array([k]),]
            cfmask += [np.arange(len(fm))[fm],]
    ###
    tuning_masks = get_tuning_masks(layer_rlist, fmaps_count)
    assert np.sum(sum(tuning_masks))==fmaps_count, "%d != %d" % (np.sum(sum(tuning_masks)), fmaps_count)    
    #layer_rlist, fmaps_sizes, fmaps_count, clmask, cfmask #, scaling
    return Torch_filter_fmaps(_fmaps_fn, clmask, cfmask), clmask, cfmask, tuning_masks
'''

