from sympy import im
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from PIL import Image
from tqdm import tqdm

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
from pathlib import Path


from torch.utils.data import Dataset

"""
    We will return a dense masks for a folder of images
    Dense masks are a 2D array of shape (H, W) where each pixel is a mask. if the pixel is 1, then the pixel is in the first mask, if the pixel is 2, then the pixel is in the second mask, etc.
    We will return a list of dense masks for each image in the folder.
"""

class FrameSegmentation:
    def __init__(self, dataset_folder: Path):
        self.image_folder = dataset_folder / 'images'
        self.mask_folder = dataset_folder / 'masks'
        if not self.mask_folder.exists():
            self.mask_folder.mkdir(parents=True)
        self.sam2 = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device='cuda', apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2,points_per_batch=16)
        self.dense_masks = []
        self.dataset = FrameSegmentationDataset(self.image_folder)
    

    def generate_dense_masks(self, image):
        masks = self.mask_generator.generate(image)
        dense_masks = np.zeros((image.shape[0], image.shape[1]))
        for i, mask in enumerate(masks):
            dense_masks[mask['segmentation']] = i+1 # 0 is background
        return dense_masks


    
    def run(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for image, image_path in tqdm(self.dataset):
            dense_masks = self.generate_dense_masks(image)
            self.dense_masks.append(dense_masks)
            self.save_dense_masks(dense_masks, image_path, save_path)
        return self.dense_masks

    
    def save_dense_masks(self, dense_masks:np.ndarray, image_path:str, save_path:str):
        dense_masks_uint8 = dense_masks.astype(np.uint8)
        png_filename = os.path.basename(image_path).replace('.jpg', '.png')
        Image.fromarray(dense_masks_uint8, mode='L').save(os.path.join(save_path, png_filename))



class FrameSegmentationDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = Path(image_folder)
        self.image_paths = list(self.image_folder.glob('*.jpg'))
    
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        return image, image_path




if __name__ == "__main__":
    frame_segmentation = FrameSegmentation('../lerf_ovs/figurines/images')
    frame_segmentation.run('../lerf_ovs/figurines/masks')






