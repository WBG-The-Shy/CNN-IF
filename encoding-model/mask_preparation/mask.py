from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi
import numpy as np

def FFA(trn_subjects,mask_root,beta_root):
    group_names = ['FFA', 'other']
    ROI_name = 'FFA'
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0]]

    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

    for k, s in enumerate(trn_subjects):
        print('--------  subject %d  -------' % s)
        voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        # voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_nsdgeneral_1.0.nii"%s)
        ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % s)
        rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % s)
        vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % s)
        voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (s))
        general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (s))
        ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz" % s)
        ###
        brain_nii_shape[s] = voxel_roi_full.shape
        print(brain_nii_shape[s])
        ###
        voxel_roi_mask_full = (voxel_roi_full > 0).flatten().astype(bool)
        voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois
        voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
        ###
        # voxel_mask[s]  = np.nan_to_num(voxel_mask_full)
        voxel_mask[s] = voxel_mask_full.flatten()
        j = 0
        for i in voxel_mask[s]:
            if i == -1:
                voxel_mask[s][j] = 0  # delete the original voxel
            j = j + 1
        voxel_mask[s] = voxel_mask[s].astype(bool)
        voxel_idx[s] = np.arange(len(voxel_mask[s]))[voxel_mask[s]]  # create index for all voxel in the voxel_mask
        # np.arange(len(voxel_mask[s])) generate index for each voxel in mask(0,1,2,3,4,...)
        voxel_roi[s] = voxel_joined_roi_full[voxel_mask[
            s]]  # after putting selected roi in kastner2015,and then finall make roi from voxel_mask in voxel_joined_roi_full
        voxel_ncsnr[s] = ncsnr_full.flatten()[voxel_mask[s]]
               print('full mask length = %d' % len(voxel_mask[s]))
        print('selection length = %d' % np.sum(voxel_mask[s]))

        for roi_mask, roi_name in iterate_roi(group, voxel_roi[s], roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))

    return brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr
def FFA_EBA_RSC_VWFA(trn_subjects,mask_root,beta_root):
    group_names = ['four_roi']
    group = [[1]]

    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

    for k, s in enumerate(trn_subjects):
        print('--------  subject %d  -------' % s)
        # voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_vcventral_1.0.nii"%s)
        ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % s)
        rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % s)
        vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % s)
        a = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        b = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        ffa_full = ffa_full.flatten()
        eba_full = eba_full.flatten()
        rsc_full = rsc_full.flatten()
        vwfa_full = vwfa_full.flatten()
        voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (s))
        general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (s))
        ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"%s)

        a = a.flatten()
        b = b.flatten()
        for i in range(len(a)):
            a[i] = 0
            b[i] = 0
        for i in range(len(a)):
            if ffa_full[i] < 1:
                ffa_full[i] = 0
            if eba_full[i] < 1:
                eba_full[i] = 0
            if rsc_full[i] < 1:
                rsc_full[i] = 0
            if vwfa_full[i] < 1:
                vwfa_full[i] = 0
        counta = 0
        countb = 0
        for i in range(len(a)):
            a[i] = ffa_full[i] + eba_full[i] + rsc_full[i] + vwfa_full[i]
            if a[i] > 0:
                counta = counta + 1
            if ffa_full[i] > 0 or eba_full[i] > 0 or rsc_full[i] > 0 or vwfa_full[i] > 0:
                b[i] = 1
                countb = countb + 1
        print('the whole voxel number is: %d' % counta)
        brain_nii_shape[s] = voxel_roi_full.shape
        print(brain_nii_shape[s])
        ###
        voxel_roi_mask_full = (a > 0).astype(bool)
        voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois

        # j = 0
        # original_num = 0   # count the original voxel num in range of roi mask
        # for i in voxel_joined_roi_full:
        #     if 0<i<=5:
        #         voxel_joined_roi_full[j] = -1   # delete the original voxel
        #         original_num = original_num + 1
        #     j = j + 1
        # print(original_num)

        voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
        ###
        voxel_mask[s] = a.flatten()
        j = 0
        for i in voxel_mask[s]:
            if i == -1:
                voxel_mask[s][j] = 0  # delete the original voxel
            j = j + 1
        voxel_mask[s] = voxel_mask[s].astype(bool)
        voxel_idx[s] = np.arange(len(voxel_mask[s]))[voxel_mask[s]]
        voxel_roi[s] = b[voxel_mask[s]]
        voxel_ncsnr[s] = ncsnr_full.flatten()[voxel_mask[s]]
       print('full mask length = %d' % len(voxel_mask[s]))
        print('selection length = %d' % np.sum(voxel_mask[s]))

        for roi_mask, roi_name in iterate_roi(group, voxel_roi[s], roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))
    return brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr

def V1_V2_V3_V4_FFA(trn_subjects,mask_root,beta_root):
    group_names = ['V1', 'V2', 'V3', 'hV4', 'FFA']
    group = [[1, 2], [3, 4], [5, 6], [7], [8]]
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

    for k, s in enumerate(trn_subjects):
        print('--------  subject %d  -------' % s)
        # load mask and roi
        visual_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % s)
        ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % s)
        rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % s)
        vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % s)
        a = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        b = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        # turn nii file into ndarry for next process
        a = a.flatten()
        b = b.flatten()
        ffa_full = ffa_full.flatten()
        eba_full = eba_full.flatten()
        rsc_full = rsc_full.flatten()
        vwfa_full = vwfa_full.flatten()
        visual_full = visual_full.flatten()
        voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (s))
        general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (s))
        ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz" % s)
        count_ffa = 0
        count_visual = 0
        count_mix = 0
        counta = 0
        # count the number of voxels and make a jointed roi
        for i in range(len(ffa_full)):
            a[i] = 0
            b[i] = 0
            if ffa_full[i] < 1:
                ffa_full[i] = 0
            if visual_full[i] < 1:
                visual_full[i] = 0
            if ffa_full[i] > 0:
                count_ffa = count_ffa + 1
                ffa_full[i] = 8
            if visual_full[i] > 0:
                count_visual = count_visual + 1
            if ffa_full[i] > 0 and visual_full[i] > 0:
                count_mix = count_mix + 1
            if visual_full[i] > 0:
                b[i] = visual_full[i]
            if ffa_full[i] > 0:
                b[i] = ffa_full[i]  # if voxel in visual roi and ffa roi think it in ffa roi
            if visual_full[i] > 0 or ffa_full[i] > 0:
                a[i] = 1
                counta = counta + 1
        print("ffa voxels = %d " % count_ffa)
        print("visual voxels = %d " % count_visual)
        print("overlap voxels = %d " % count_mix)
        print("joint roi voxels = %d " % counta)
        brain_nii_shape[s] = voxel_roi_full.shape
        print(brain_nii_shape[s])
        ###
        voxel_roi_mask_full = (a > 0).astype(bool)
        voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois

        voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
        ###
        voxel_mask[s] = a.flatten()
        voxel_mask[s] = voxel_mask[s].astype(bool)
        voxel_idx[s] = np.arange(len(voxel_mask[s]))[voxel_mask[s]]
        voxel_roi[s] = b[voxel_mask[s]]
        voxel_ncsnr[s] = ncsnr_full.flatten()[voxel_mask[s]]

        print('full mask length = %d' % len(voxel_mask[s]))
        print('selection length = %d' % np.sum(voxel_mask[s]))

        for roi_mask, roi_name in iterate_roi(group, voxel_roi[s], roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))
    return brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr

def V1_V2_V3_V4(trn_subjects,mask_root,beta_root):
    group_names = ['V1', 'V2', 'V3', 'hV4']
    group = [[1, 2], [3, 4], [5, 6], [7]]
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

    for k, s in enumerate(trn_subjects):
        print('--------  subject %d  -------' % s)
        # load mask and roi
        visual_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % s)
        ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % s)
        rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % s)
        vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % s)
        a = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        b = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        # turn nii file into ndarry for next process
        a = a.flatten()
        b = b.flatten()
        ffa_full = ffa_full.flatten()
        eba_full = eba_full.flatten()
        rsc_full = rsc_full.flatten()
        vwfa_full = vwfa_full.flatten()
        visual_full = visual_full.flatten()
        voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (s))
        general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (s))
        ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz" % s)
        count_ffa = 0
        count_visual = 0
        count_mix = 0
        counta = 0
        # count the number of voxels and make a jointed roi
        for i in range(len(ffa_full)):
            a[i] = 0
            b[i] = 0
            # if ffa_full[i] < 1:
            #     ffa_full[i] = 0
            if visual_full[i] < 1:
                visual_full[i] = 0
            # if ffa_full[i] > 0:
            #     count_ffa = count_ffa + 1
            #     ffa_full[i] = 8
            if visual_full[i] > 0:
                count_visual = count_visual + 1
            # if ffa_full[i] > 0 and visual_full[i] > 0:
            #     count_mix = count_mix + 1
            if visual_full[i] > 0:
                b[i] = visual_full[i]
                a[i] = 1
            # if ffa_full[i] > 0:
            #     b[i] = ffa_full[i]  # if voxel in visual roi and ffa roi think it in ffa roi
            # if visual_full[i] > 0 or ffa_full[i] > 0:
            #     a[i] = 1
            #     counta = counta + 1
        # print("ffa voxels = %d " % count_ffa)
        print("visual voxels = %d " % count_visual)
        # print("overlap voxels = %d " % count_mix)
        # print("joint roi voxels = %d " % counta)
        brain_nii_shape[s] = voxel_roi_full.shape
        print(brain_nii_shape[s])
        ###
        voxel_roi_mask_full = (a > 0).astype(bool)
        voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois

        voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
        ###
        voxel_mask[s] = a.flatten()
        voxel_mask[s] = voxel_mask[s].astype(bool)
        voxel_idx[s] = np.arange(len(voxel_mask[s]))[voxel_mask[s]]
        voxel_roi[s] = b[voxel_mask[s]]
        voxel_ncsnr[s] = ncsnr_full.flatten()[voxel_mask[s]]

        print('full mask length = %d' % len(voxel_mask[s]))
        print('selection length = %d' % np.sum(voxel_mask[s]))

        for roi_mask, roi_name in iterate_roi(group, voxel_roi[s], roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))
    return brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr

def V1_V2_V3_V4_FFA_EBA_RSC_VWFA(trn_subjects,mask_root,beta_root):
    group_names = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA','RSC','VWFA']
    group = [[1], [2], [3], [4], [5], [6], [7], [8]]
    brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr = {}, {}, {}, {}, {}

    for k, s in enumerate(trn_subjects):
        print('--------  subject %d  -------' % s)
        # load mask and roi
        visual_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % s)
        ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz" % s)
        rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz" % s)
        vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz" % s)
        a = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        b = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        # turn nii file into ndarry for next process
        a = a.flatten()
        b = b.flatten()
        ffa_full = ffa_full.flatten()
        eba_full = eba_full.flatten()
        rsc_full = rsc_full.flatten()
        vwfa_full = vwfa_full.flatten()
        visual_full = visual_full.flatten()
        voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_roi_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz" % s)
        voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (s))
        general_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (s))
        ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz" % s)
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
        brain_nii_shape[s] = voxel_roi_full.shape
        print(brain_nii_shape[s])
        ###
        voxel_roi_mask_full = (a > 0).astype(bool)
        voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois

        voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[
            voxel_roi_mask_full]  # overwrite with prf rois
        ###
        voxel_mask[s] = a.flatten()
        voxel_mask[s] = voxel_mask[s].astype(bool)
        voxel_idx[s] = np.arange(len(voxel_mask[s]))[voxel_mask[s]]
        voxel_roi[s] = b[voxel_mask[s]]
        voxel_ncsnr[s] = ncsnr_full.flatten()[voxel_mask[s]]

        print('full mask length = %d' % len(voxel_mask[s]))
        print('selection length = %d' % np.sum(voxel_mask[s]))

        for roi_mask, roi_name in iterate_roi(group, voxel_roi[s], roi_map, group_name=group_names):
            print("%d \t: %s" % (np.sum(roi_mask), roi_name))
    return brain_nii_shape, voxel_mask, voxel_idx, voxel_roi, voxel_ncsnr