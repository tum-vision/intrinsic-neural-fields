import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import imageio
import json

from cameras import DistortionTypes
from mesh import MeshViewPreProcessor
from utils import load_obj_mask_as_tensor, load_depth_as_numpy, load_cameras
from dataset import load_meshroom_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("out_dir", type=str, help="Path to the directory where the preprocessed data should be stored")
    parser.add_argument("path_to_mesh", type=str, help="Path to the mesh file")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset which is preprocessed")
    parser.add_argument("split", type=str, help="Dataset split")
    parser.add_argument("--dataset_type", type=str, help="Dataset Type", default=None)
    parser.add_argument("--check_depth", action="store_true", help="Will check the depth from ray-mesh intersection with the depth map of the dataset.")
    args = parser.parse_args()
    return args


def preprocess_views(mesh_view_pre_proc, mesh_views_list, dataset_path, check_depth=False):
    for mesh_view in tqdm(mesh_views_list):
        mesh_view_path = os.path.join(dataset_path, mesh_view)

        camCv2world, K = load_cameras(mesh_view_path)

        # Load depth map for building a mask
        obj_mask = load_obj_mask_as_tensor(mesh_view_path)

        # Load image
        img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))
        img = torch.from_numpy(img).to(dtype=torch.float32)
        img /= 255.

        # Preprocess and cache the current view
        if check_depth:
            depth = load_depth_as_numpy(mesh_view_path)
            assert np.all(np.isfinite(depth))
            assert np.all(depth > 0)
            assert np.all(depth <= 1e10)
            mesh_view_pre_proc.cache_single_view(camCv2world, K, obj_mask, img, depth_check=depth)
        else:
            mesh_view_pre_proc.cache_single_view(camCv2world, K, obj_mask, img)

    mesh_view_pre_proc.write_to_disk()


def preprocess_meshroom_data(mesh_view_pre_proc, metadata, dataset_path):
    # Get H and W
    img = imageio.imread(os.path.join(dataset_path, metadata["views"][0]["view_file"]))
    H, W = img.shape[:2]

    K = torch.from_numpy(np.array(metadata["K"]).astype(np.float32))
    distortion_params = list(map(float, metadata["distortion_params"]))
    distortion_type = "meshroom_radial_k3"

    for view_data in tqdm(metadata["views"]):
        # Load view image
        img = imageio.imread(os.path.join(dataset_path, view_data["view_file"])) / 255.
        img = torch.from_numpy(img).to(torch.float32)

        # Get background mask
        obj_mask = np.load(os.path.join(dataset_path, view_data["obj_mask_file"]))
        bg_mask = obj_mask == False

        # Ensure that we have a white background
        img[bg_mask] = 1

        # Get cam2world
        cam2world = np.array(view_data["cam2world"]).astype(np.float32)
        cam2world = torch.from_numpy(cam2world)
        cam2world = cam2world[:3]

        # Preprocess and cache the current view
        mesh_view_pre_proc.cache_single_view(cam2world, 
                                             K, 
                                             obj_mask, 
                                             img, 
                                             distortion_coeffs=distortion_params, 
                                             distortion_type=DistortionTypes.MESHROOM_RADIAL_K3)

    mesh_view_pre_proc.write_to_disk()


def preprocess_dataset(split, dataset_path, path_to_mesh, out_dir, dataset_type, check_depth):
    split_out_dir = os.path.join(out_dir, split)

    if os.path.exists(split_out_dir):
        raise RuntimeError(f"Error: You are trying to overwrite the following directory: {split_out_dir}")
    os.makedirs(split_out_dir, exist_ok=True)

    mesh_view_pre_proc = MeshViewPreProcessor(path_to_mesh, split_out_dir)

    if dataset_type is None:
        with open(os.path.join(dataset_path, f"{split}.lst"), "r") as file_handle:
            mesh_views_list = [line[:-1] if line.endswith('\n') else line for line in file_handle.readlines()]
        preprocess_views(mesh_view_pre_proc, mesh_views_list, dataset_path, check_depth=check_depth)
    elif dataset_type == "meshroom_radial_k3":
        metadata = load_meshroom_metadata(dataset_path, split)
        preprocess_meshroom_data(mesh_view_pre_proc, metadata, dataset_path)
    else:
        raise NotImplementedError(f"Unknown dataset type: {type}")


def main():
    args = parse_args()
    print("Preprocessing dataset...")
    preprocess_dataset(args.split, args.dataset_path, args.path_to_mesh, args.out_dir, args.dataset_type, args.check_depth)


if __name__ == "__main__":
    main()
