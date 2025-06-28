import numpy as np


import torch
import torch.nn.functional as F
from pathlib import Path    
from tqdm import tqdm

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="language_features_test")
    parser.add_argument("--output_dir", type=str, default="features")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    segment_list = list(input_dir.glob("*s.npy"))
    segment_list.sort()
    feature_list = list(input_dir.glob("*f.npy"))
    feature_list.sort()
    feature_name_list = list(output_dir.glob("*.pt"))
    feature_name_list.sort()

    for segment_path, feature_path, feature_name in tqdm(zip(segment_list, feature_list, feature_name_list), total=len(segment_list)):
        mask = torch.from_numpy(np.load(segment_path)).cuda().long() + 1
        features = torch.from_numpy(np.load(feature_path)).cuda()
        zero_row     = torch.zeros(1, 512, device=features.device, dtype=features.dtype)
        features_pad = torch.cat([zero_row, features], dim=0) 
        feat_map = F.embedding(mask, features_pad).sum(dim=0)
        feat_map = feat_map / (feat_map.norm(dim=-1, keepdim=True) + 1e-6)

        torch.save(feat_map, feature_name)        

