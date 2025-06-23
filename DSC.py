import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def load_nrrd_image(file_path, start_index=0, end_index=None):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)  # numpy
    
    data = data.transpose(2,1,0)  # Convert to (H, W, D) format

    if end_index is not None:
        data = data[:, :, start_index:end_index]
    
    return data

def load_nib_image(file_path,start_index=0, end_index=None, type=None):
    img = nib.load(file_path)
    data = img.get_fdata()  # numpy
    #affine = img.affine  
    if end_index is not None:
        data = data[:, :, start_index:end_index]

    if type is not None:
        data = (data == type).astype(np.uint8)  # only specific type

    return data

def dsc_score(pred_mask, gt_mask):
    pred_bin = (pred_mask>0.5).astype(np.uint8)
    gt_bin = (gt_mask>0.5).astype(np.uint8)
    
    intersection = np.logical_and(pred_bin, gt_bin)
    #intersection = np.sum(pred_bin * gt_bin)  # Count overlapping pixels
    sc = 2. * intersection / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)
    dsc = 2. * intersection.sum() / (pred_bin.sum() + gt_bin.sum() + 1e-8)
    
    return dsc

def main():
    start_index = 34
    end_index = 71
    type = 2

    pred_path = "../archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/pred_mask.seg.nrrd"  # seg
    gt_path = "../archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"  # flair seg
    flair_path = "../archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"  # flair seg

    # Load
    img = load_nib_image(flair_path)  # 45:95
    pred_mask = load_nrrd_image(pred_path,start_index=start_index,end_index=end_index)  # 45:95
    #gt_mask = load_nib_image(gt_path, start_index=start_index, end_index=end_index,type=type)  # Load first 10 slices
    gt_mask_1 = load_nib_image(gt_path, start_index=start_index, end_index=end_index,type=1)  
    gt_mask_2 = load_nib_image(gt_path, start_index=start_index, end_index=end_index,type=2)
    gt_mask_4 = load_nib_image(gt_path, start_index=start_index, end_index=end_index,type=4)
    gt_mask = gt_mask_1 + gt_mask_2 + gt_mask_4  # Combine all types


    dsc = dsc_score(pred_mask, gt_mask_1)
    print(f"DSC Score: {dsc}")

if __name__ == "__main__":
    main()
