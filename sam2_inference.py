import os
import torch
import numpy as np
import nibabel as nib
from sam2_train.build_sam import build_sam2
from sam2_train.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
# mps
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(f"Using device: {device}")


def load_nib_image(file_path):
    img = nib.load(file_path)
    data = img.get_fdata() # (240,240,155)
    affine = img.affine  # Get the affine transformation matrix
    #print(f"Image shape: {data.shape}")  # Print the shape of the image data
    return data, affine

def get_slice(data, slice_index):
    # slice
    image_slice = data[:, :, slice_index]  # (240, 240)

    # norm
    norm_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()+1e-8)  # Normalize to [0, 1]
    rgb_slice = np.stack([norm_slice] * 3, axis=-1)  # RGB 240,240, 3
    rgb_slice = (rgb_slice * 255).astype(np.uint8)  # Convert to uint8

    return  rgb_slice

def main():
    sample_path = "../archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"  # flair seg
    # Load image
    data, affine = load_nib_image(sample_path) # 240, 240, 3

    # model
    model = build_sam2(
        config_file="sam2_hiera_s.yaml",
        ckpt_path="./checkpoints/sam2_hiera_small.pt",
        device='cpu',
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
    )

    #model = model.to(device)

    predictor = SAM2ImagePredictor(sam_model=model)

    # empty mask
    mask_volume = np.zeros(data.shape, dtype=np.uint8)  # Initialize an empty mask volume

    # get slices
    start_index = 45
    end_index = 71
    interval = 5
    save = True
    point_coords = np.array([[112, 83],[109,86],[109,86],[109,86],[109,86],[121,79]])  # point in the middle
    set = 0

    for slice_index in range(start_index, end_index, interval):  # sample slices
        slice = get_slice(data, slice_index)

        predictor.set_image(slice)
        #point_coords = np.array([[92, 97.5],[158,82],[119,111],[173,89.4]])  # point in the middle
        #point_labels = np.array([1,1,1,1]) # front
        #point_coords = np.array([[109, 86]])  # point in the middle
        point_labels = np.array([1])
        box = np.array([[70, 150, 100, 110]])  # box around the point
        # inference
        masks, quality, logits = predictor.predict(point_coords=point_coords[set].reshape(1,2),point_labels=point_labels,box=box,multimask_output=False)
        set += 1

        # set mask
        mask_volume[:, :, slice_index] = masks[0].astype(np.uint8) 

    # # visualize
    # plt.imshow(slice,cmap='gray')  # original image
    # plt.imshow(masks[0], alpha=0.5, cmap='Reds')  # Overlay the mask on the image
    # #plt.imshow(masks[1], alpha=0.5, cmap='Blues')  # Overlay the second mask on the image
    # plt.title(f"Mask Quality: {quality[0]:.2f}")
    # plt.show()

    # save mask volume
    if save:
        mask_nifti = nib.Nifti1Image(mask_volume, affine)
        output_path = "../output_mask.nii"
        nib.save(mask_nifti, output_path)
        print(f"Mask volume saved to {output_path}")


if __name__ == "__main__":
    main()
    # You can now use the model for inference or further training
    # For example, to run inference on a sample input:
    # sample_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust shape as needed
    # output = model(sample_input)
    # print(f"Output shape: {output.shape}")