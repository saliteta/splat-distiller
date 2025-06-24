import torch
from PIL import Image
import open_clip
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        size=(378, 378),
        interpolation=InterpolationMode.BICUBIC,  # or Image.BICUBIC
        antialias=True
    ),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])


model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
model.eval() 
model.to('cuda')
tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')


class FrameFeature:
    def __init__(self, dataset_folder: Path):
        self.image_folder = dataset_folder / 'images'
        self.mask_folder = dataset_folder / 'masks'
        self.feature_folder = dataset_folder / 'features_samclip'
        if not self.feature_folder.exists():
            self.feature_folder.mkdir(parents=True)
        self.image_paths = list(self.image_folder.glob('*.jpg'))
        self.mask_paths = list(self.mask_folder.glob('*.png'))

    def run(self):
        dataset = FrameFeatureDataset(self.image_folder, self.mask_folder)
        for masked_images, image_path in tqdm(dataset):
            with torch.no_grad():
                masked_images = masked_images.to('cuda')
                image_features = model.encode_image(masked_images)[:, :512]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                torch.save(image_features, self.feature_folder / f'{image_path.stem}.pt')


class FrameFeatureDataset(Dataset):
    def __init__(self, image_folder: Path, mask_folder: Path):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_paths = list(self.image_folder.glob('*.jpg'))
        self.mask_paths = list(self.mask_folder.glob('*.png'))
        self.preprocess = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        masked_images = self.get_masked_image(image, mask)
        masked_images = torch.stack([self.preprocess(masked_image) for masked_image in masked_images])
        return masked_images, image_path


    def get_masked_image(self, image: Image.Image, mask: Image.Image):
        image_array = np.array(image.convert("RGB"))
        mask_array = np.array(mask.convert("L"))
        masks = ((np.arange(mask_array.max())+1)[:, None, None] == mask_array[None, :, :])
        masked_images = masks[..., None] * image_array[None, ...]
        return masked_images
    
