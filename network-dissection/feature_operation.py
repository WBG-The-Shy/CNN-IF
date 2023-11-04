import csv
import os
from torch.autograd import Variable as V  #zi?pytorch??Variable??????????????????????????????????????????????????from scipy.misc import imresize
from PIL import Image

from torch.nn.parameter import Parameter
import numpy as np
import torch
import settings
import time
import util.upsample as upsample
import util.vecquantile as vecquantile
import multiprocessing.pool as pool
from loader.data_loader import load_csv
from loader.data_loader import SegmentationData, SegmentationPrefetcher
from util.rotate import randomRotationPowers
device = torch.device("cuda:0")
features_blobs = []  #? hook the feature extractor
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'))  #add the result file path
        self.data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
        self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
        self.mean = [109.5388,118.6897,124.6901]
        dims = 256
        seed = 1
        alpha = np.arange(0.1, 1.0 + 1e-15, 0.1)
        self.rots = np.ones([dims,dims])




    def feature_extraction(self, model=None, memmap=True):
        loader = self.loader # the wat to load dataset have finished   so it can accerlate the time
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(zip(mmap_files,mmap_max_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=float,mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=float, mode='r', shape=tuple(features_size[i][:2]))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        weights = np.loadtxt(
            settings.root_dir +
            "voxel_weights",
            dtype=np.float32, delimiter=",")
        index = np.loadtxt(
            "voxel_index",
            dtype=np.float32, delimiter=",")
        indexx = np.zeros(len(index), dtype=bool)
        for i in range(len(index)):
            if index[i] > 0:
                indexx[i] = True
        weights = weights[indexx][:]

        weight = list(torch.split(torch.from_numpy(weights), [64, 192, 384, 256, 256, 512, 512, 512], dim=1)) #split weights into different size of feature
        for batch_idx,batch in enumerate(loader.tensor_batches(bgr_mean=self.mean)): #tensor_batches:Returns a single batch as an array of tensors, one per category
            if batch_idx == 494:
                print('the last one')
            del features_blobs[:]
            input = batch[0]
            batch_size = len(input)
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))
            input = torch.from_numpy(input[:, ::-1, :, :].copy())
            input.div_(255.0) #pass = do nothing  normalization
            if settings.GPU:
                input = input.to(device)
            with torch.no_grad():
                input_var = V(input)
            logit = model.forward(input_var)


            feature = features_blobs[13]
            features_blob = []
            # feature = torch.tensordot(torch.from_numpy(feature), weight[4], dims=[[1], [1]])
            # feature = torch.transpose(torch.transpose(feature, 2, 3), 1, 2)
            feature = torch.from_numpy(feature)
            features_blob.append(feature)

            #gnet
            # feature1 = a2[0]
            # feature2 = a2[1]
            # feature1 = feature1.cpu().detach().numpy()  # feature1,2 is on GPU then put them into numpy in cpu
            # feature2 = feature2.cpu().detach().numpy()
            #
            # features_blob = []
            # feature1 = torch.tensordot(torch.from_numpy(feature1), weight[0], dims=[[1], [1]])
            # feature2 = torch.tensordot(torch.from_numpy(feature2), weight[1], dims=[[1], [1]])
            # feature1 = torch.transpose(torch.transpose(feature1, 2, 3), 1, 2)
            # feature2 = torch.transpose(torch.transpose(feature2, 2, 3), 1, 2)
            # features_blob.append(features_blobs[0])
            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blob):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap(mmap_max_files[i],dtype=float,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)
            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blob):
                    size_features = (
                    len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=float, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
            np.save(features_size_file, features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blob):


                feat_batch = feat_batch.cpu().detach().numpy()

                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch
        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures
        return wholefeatures,maxfeatures

    def quantile_threshold(self, features, savepath=''):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            return np.load(qtpath)
        print("calculating quantile threshold")
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        start_time = time.time()
        last_batch_time = start_time
        batch_size = 4096
        for i in range(0, features.shape[0], batch_size):
            batch_time = time.time()  #the time is begin from 1970
            rate = i / (batch_time - start_time + 1e-15)
            batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time
            print('Processing quantile index %d: %f %f' % (i, rate, batch_rate))
            batch = features[i:i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(-1, features.shape[1]) #reshape into features,batch*fmsize*fmsize
            quant.add(batch)
        ret = quant.readout(1000)[:, int(1000 * (1-settings.QUANTILE)-1)]  #devide into 1000 dots to sample data
        if savepath:
            np.save(qtpath, ret)
        return ret
        # return np.percentile(features,100*(1 - settings.QUANTILE),axis=axis)

    @staticmethod
    def tally_job(args):
        features, data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, start, end = args
        units = features.shape[1]
        size_RF = (settings.IMG_SIZE / features.shape[2], settings.IMG_SIZE / features.shape[3])
        fieldmap = ((0, 0), size_RF, size_RF)
        pd = SegmentationPrefetcher(data, categories=data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD, start=start, end=end)
        count = start
        start_time = time.time()
        last_batch_time = start_time
        for batch in pd.batches():
            batch_time = time.time()
            rate = (count - start) / (batch_time - start_time + 1e-15)
            batch_rate = len(batch) / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            print('labelprobe image index %d, items per sec %.4f, %.4f' % (count, rate, batch_rate))
            # each image have a concept_map to compute iou with input x
            for concept_map in batch:
                count += 1
                # if count == 6610:
                #     print(count)
                img_index = concept_map['i'] # get the index or the order of image
                scalars, pixels = [], []
                for cat in data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        # label_group = label_group.ravel()
                        # for i in range(len(label_group)):
                        #     if label_group[i]>0:
                        #         label_group[i] = pd.segmentation.category_unmap[cat][label_group[i]-1]
                        pixels.append(label_group)
                for scalar in scalars:
                    tally_labels[scalar] += concept_map['sh'] * concept_map['sw']
                if pixels:
                    pixels = np.concatenate(pixels)
                    tally_label = np.bincount(pixels.ravel())
                    # np.bincount to get the frequence of num from min to max
                    # np.ravel = np.flatten  #ravel ?????????? flatten ??????? ????????????copy????????view?? numpy.ravel() ???????????????numpy.flatten() ????????????????????????#
                    if len(tally_label) > 0:
                        tally_label = np.delete(tally_label,0)
                        # tally_label[0] = 0  # remove the 0 data from tally_label becaurse label num is start from number:1
                    tally_labels[:len(tally_label)] += tally_label # add all label from tally_label into tally_labels
                # now we get one image all labels in its pixel     it is recorded in tally_labels
                for unit_id in range(units):
                    feature_map = features[img_index][unit_id]
                    if feature_map.max() > threshold[unit_id]:  # mode='F'   32bit float pixel
                        mask = np.array(Image.fromarray(feature_map, mode='F').resize((concept_map['sh'], concept_map['sw']))) #F mean grey image   2 binary segmentation
                        # mask = imresize(feature_map, (concept_map['sh'], concept_map['sw']), mode='F')   resample = 0 binear
                        #reduction = int(round(settings.IMG_SIZE / float(concept_map['sh'])))      upsample into new size
                        #mask = upsample.upsampleL(fieldmap, feature_map, shape=(concept_map['sh'], concept_map['sw']), reduction=reduction)
                        indexes = np.argwhere(mask > threshold[unit_id])   # return  Array a ?????1?????
                        # the length of indexes is the pixels num which exceed threshold[unit]
                        tally_units[unit_id] += len(indexes)  # write the len of index(pixel num) which is exceed threshold in unit_id
                        if len(pixels) > 0: # select all pixels in image in 113*113 size   #indexs[:,0] = the first line (6509x1)
                            tally_bt = np.bincount(pixels[:, indexes[:, 0], indexes[:, 1]].ravel())   # the pixel is all segmentation map which is concatene together
                            if len(tally_bt) > 0:
                                tally_bt = np.delete(tally_bt,0)
                            tally_cat = np.dot(tally_bt[None,:], data.labelcat[:len(tally_bt), :])[0]
                            tally_both[unit_id,:len(tally_bt)] += tally_bt
                        for scalar in scalars:
                            tally_cat += data.labelcat[scalar]
                            tally_both[unit_id, scalar] += len(indexes)
                        tally_units_cat[unit_id] += len(indexes) * (tally_cat > 0)


    def tally(self, features, threshold, savepath=''):
        csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(csvpath):
            return load_csv(csvpath)

        units = features.shape[1]
        labels = len(self.data.label) # the number of labels in all image from Broden datasets
        categories = self.data.category_names()
        tally_both = np.zeros((units,labels),dtype=np.float64)
        tally_units = np.zeros(units,dtype=np.float64)
        tally_units_cat = np.zeros((units,len(categories)), dtype=np.float64)
        tally_labels = np.zeros(labels,dtype=np.float64)

        if settings.PARALLEL > 1:
            psize = int(np.ceil(float(self.data.size()) / settings.PARALLEL))
            ranges = [(s, min(self.data.size(), s + psize)) for s in range(0, self.data.size(), psize) if
                      s < self.data.size()]
            params = [(features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both) + r for r in ranges]
            threadpool = pool.ThreadPool(processes=settings.PARALLEL)
            threadpool.map(FeatureOperator.tally_job, params)
        else:
            FeatureOperator.tally_job((features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, 0, self.data.size()))


        primary_categories = self.data.primary_categories_per_index()
        # print(tally_units_cat)
        a = tally_units_cat
        tally_units_cat = np.dot(tally_units_cat, self.data.labelcat.T)  # self.data.labelcat.T   transpose the aixs
        # print(tally_units_cat)
        iou = tally_both / (tally_units_cat + tally_labels[np.newaxis,:] - tally_both + 1e-10)
        pciou = np.array([iou * (primary_categories[np.arange(iou.shape[1])] == ci)[np.newaxis, :] for ci in range(len(self.data.category_names()))])
        label_pciou = pciou.argmax(axis=2) # get the max index in specific axis
        name_pciou = [
            [self.data.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))]
        score_pciou = pciou[
            np.arange(pciou.shape[0])[:, np.newaxis],
            np.arange(pciou.shape[1])[np.newaxis, :],
            label_pciou]
        bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
        ordering = score_pciou.max(axis=0).argsort()[::-1]
        rets = [None] * len(ordering)


        for i,unit in enumerate(ordering):
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            data = {
                'unit': (unit + 1),
                'category': categories[bestcat],
                'label': name_pciou[bestcat][unit],
                'score': score_pciou[bestcat][unit]
            }
            for ci, cat in enumerate(categories):
                label = label_pciou[ci][unit]
                data.update({
                    '%s-label' % cat: name_pciou[ci][unit],
                    '%s-truth' % cat: tally_labels[label],
                    '%s-activation' % cat: tally_units_cat[unit, label],
                    '%s-intersect' % cat: tally_both[unit, label],
                    '%s-iou' % cat: score_pciou[ci][unit]
                })
            rets[i] = data

        if savepath:
            import csv
            csv_fields = sum([[
                '%s-label' % cat,
                '%s-truth' % cat,
                '%s-activation' % cat,
                '%s-intersect' % cat,
                '%s-iou' % cat] for cat in categories],
                ['unit', 'category', 'label', 'score'])
            with open(csvpath, 'w') as f:
                writer = csv.DictWriter(f, csv_fields)
                writer.writeheader()
                for i in range(len(ordering)):
                    writer.writerow(rets[i])
        return rets
