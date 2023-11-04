group_names = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST','FFA', 'EBA', 'RSC', 'VWFA', 'other'] #divide voxel to different group
group = [[1,2],[3,4],[5,6], [7], [16, 17], [14, 15], [18,19,20,21,22,23], [8, 9], [10,11], [13], [12], [26], [27], [28], [29], [24,25,0]]  # mapping to roi map
voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%subject) #
voxel_roi_full  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%subject) # roi visual
voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz"%(subject))
general_mask_full  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz"%(subject))
ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"%subject)
ffa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz"%(subject))
eba_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz"%(subject))
rsc_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz"%(subject))
vwfa_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz"%(subject))
brain_nii_shape = voxel_roi_full.shape
print (brain_nii_shape)    #output the size of roi
ffa_full = ffa_full.flatten()
eba_full = eba_full.flatten()
rsc_full = rsc_full.flatten()
vwfa_full = vwfa_full.flatten()
voxel_kast_full = voxel_kast_full.flatten()
voxel_roi_full = voxel_roi_full.flatten()
voxel_jointed_roi = np.zeros(len(voxel_kast_full),dtype=np.float64)
for i in range(len(voxel_roi_full)):
    if voxel_kast_full[i]>0:
        voxel_jointed_roi[i] = voxel_kast_full[i]
    if voxel_roi_full[i]>0:
        voxel_jointed_roi[i] = voxel_roi_full[i]
    if ffa_full[i]>0:
        voxel_jointed_roi[i] = 26
    if eba_full[i]>0:
        voxel_jointed_roi[i] = 27
    if rsc_full[i]>0:
        voxel_jointed_roi[i] = 28
    if vwfa_full[i]>0:
        voxel_jointed_roi[i] = 29
#noise ceiling snr
###
###
count1 = 0
for i in voxel_roi_full.flatten():
    if i>0:
        count1 = count1 + 1
print("the infect voxel numbers = %d"%count1)

voxel_mask  = np.nan_to_num(voxel_mask_full).flatten().astype(bool)     #np.flatten  into a 1 dim array
# print(voxel_mask)   len(voxel_mask)=699192    the voxel num is 699192
voxel_idx   = np.arange(len(voxel_mask))[voxel_mask]   #np.arrange  return a average num from start to end step = default
# print(voxel_idx)
voxel_roi   = voxel_jointed_roi[voxel_mask]
# print(voxel_roi)
voxel_ncsnr = ncsnr_full.flatten()[voxel_mask]
# print(voxel_ncsnr)
print ('full mask length = %d'%len(voxel_mask))   #all voxels
print ('selection length = %d'%np.sum(voxel_mask)) # select voxels    the number of true = the number of voxel in roi

for roi_mask, roi_name in iterate_roi(group, voxel_roi, roi_map, group_name=group_names):
    print ("%d \t: %s" % (np.sum(roi_mask), roi_name))   #devide roi mask and named