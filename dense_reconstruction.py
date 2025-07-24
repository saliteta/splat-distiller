#!/usr/bin/env python3
"""
Simple COLMAP dense reconstruction script
Takes sparse/0 results and creates dense point cloud and mesh using default parameters
"""

import os
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_command(cmd, description):
    logging.info(f"Running: {description}")
    print(f"Command: {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"{description} failed with exit code {exit_code}")
        exit(exit_code)
    logging.info(f"{description} completed successfully")

def main():
    parser = ArgumentParser("COLMAP Dense Reconstruction")
    parser.add_argument("--source_path", "-s", required=True, type=str,
                       help="Path to dataset with sparse/0 reconstruction")
    parser.add_argument("--colmap_executable", default="", type=str,
                       help="Path to COLMAP executable")
    parser.add_argument("--no_gpu", action="store_true", 
                       help="Disable GPU acceleration")
    parser.add_argument("--skip_mesh", action="store_true",
                       help="Skip meshing step (only create dense point cloud)")
    
    args = parser.parse_args()
    setup_logging()
    
    # Setup paths
    source_path = Path(args.source_path)
    sparse_path = source_path / "sparse" / "0"
    images_path = source_path / "images"
    dense_path = source_path / "dense"
    
    # Verify sparse reconstruction exists
    if not sparse_path.exists():
        logging.error(f"Sparse reconstruction not found at {sparse_path}")
        logging.error("Please run convert.py first to create sparse reconstruction")
        exit(1)
    
    if not images_path.exists():
        logging.error(f"Images directory not found at {images_path}")
        exit(1)
    
    # Setup COLMAP command
    colmap_cmd = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
    
    # Create dense directory
    dense_path.mkdir(exist_ok=True)
    logging.info(f"Creating dense reconstruction in {dense_path}")
    
    # Step 1: Image undistortion - output directly to dense workspace
    dense_sparse_path = dense_path / "sparse"
    dense_images_path = dense_path / "images"
    
    if not dense_images_path.exists():
        logging.info("Step 1: Image undistortion")
        img_undist_cmd = (
            f"{colmap_cmd} image_undistorter "
            f"--image_path {images_path} "
            f"--input_path {sparse_path} "
            f"--output_path {dense_path} "
            f"--output_type COLMAP"
        )
        run_command(img_undist_cmd, "Image undistortion")
    else:
        logging.info("Step 1: Using existing undistorted images in dense workspace")
    
    # Create stereo directory and configuration files
    stereo_path = dense_path / "stereo"
    stereo_path.mkdir(exist_ok=True)
    
    # Get list of images and create configuration files
    images_list = []
    for img_file in (dense_images_path).glob("*.jpg"):
        images_list.append(img_file.name)
    for img_file in (dense_images_path).glob("*.png"):
        images_list.append(img_file.name)
    
    # Create patch-match.cfg
    patch_match_cfg = stereo_path / "patch-match.cfg"
    with open(patch_match_cfg, 'w') as f:
        for img in images_list:
            f.write(f"{img}\n__auto__, 20\n")
    
    # Create fusion.cfg
    fusion_cfg = stereo_path / "fusion.cfg"
    with open(fusion_cfg, 'w') as f:
        for img in images_list:
            f.write(f"{img}\n")
    
    logging.info(f"Created configuration files with {len(images_list)} images")
    
    # Step 2: Patch match stereo
    logging.info("Step 2: Patch match stereo")
    gpu_flag = "--PatchMatchStereo.gpu_index -1" if args.no_gpu else ""
    patch_match_cmd = (
        f"{colmap_cmd} patch_match_stereo "
        f"--workspace_path {dense_path} "
        f"--workspace_format COLMAP "
        f"--PatchMatchStereo.geom_consistency 1 "
        f"--PatchMatchStereo.max_image_size 800 "  
        f"{gpu_flag}"
    )
    run_command(patch_match_cmd, "Patch match stereo")
    
    # Step 3: Stereo fusion  
    logging.info("Step 3: Stereo fusion")
    stereo_fusion_cmd = (
        f"{colmap_cmd} stereo_fusion "
        f"--workspace_path {dense_path} "
        f"--workspace_format COLMAP "
        f"--input_type geometric "
        f"--output_path {dense_path / 'fused.ply'}"
    )
    run_command(stereo_fusion_cmd, "Stereo fusion")
    
    # Step 4: Poisson meshing (optional)
    if not args.skip_mesh:
        logging.info("Step 4: Poisson meshing")
        poisson_cmd = (
            f"{colmap_cmd} poisson_mesher "
            f"--input_path {dense_path / 'fused.ply'} "
            f"--output_path {dense_path / 'meshed-poisson.ply'}"
        )
        run_command(poisson_cmd, "Poisson meshing")
    
    logging.info("Dense reconstruction completed!")
    logging.info(f"Dense point cloud: {dense_path / 'fused.ply'}")
    if not args.skip_mesh:
        logging.info(f"Poisson mesh: {dense_path / 'meshed-poisson.ply'}")

if __name__ == "__main__":
    main()