import h5py
import numpy as np
from os.path import join
import yaml
from functools import lru_cache
import netpbmfile
import cv2
import open3d as o3d


@lru_cache(maxsize=128)
def read_meta(in_dir):
    with open(join(in_dir, "metadata.yaml"), "r") as fp:
        return yaml.safe_load(fp)


def read_depth(in_dir, cam, deg, dtype=np.float):
    assert cam.startswith("NP")
    meta = read_meta(in_dir)
    with h5py.File(join(in_dir, f"{cam}_{deg}.h5"), "r") as f:
        depth = f.get("depth")[()]

    return depth.astype(dtype) * meta['depth_scale_factor']


def read_mask(in_dir, cam, deg, dtype=np.uint8):
    return netpbmfile.imread(join(in_dir, "masks", f"{cam}_{deg}_mask.pbm")).astype(dtype)


def read_img(in_dir, cam, deg):
    return cv2.cvtColor(cv2.imread(join(in_dir, f"{cam}_{deg}.jpg"), -1), cv2.COLOR_BGR2RGB)


def read_raw(in_dir, cam, deg, depth_dtype=np.float, mask_dtype=np.uint8):
    img = read_img(in_dir, cam, deg)
    depth = read_depth(in_dir, cam, deg, dtype=depth_dtype)
    mask = read_mask(in_dir, cam, deg, dtype=mask_dtype)
    return img, depth, mask


def read_cloud(in_dir, cam, deg):
    assert cam.startswith("NP")
    return o3d.io.read_point_cloud(join(in_dir, "clouds", f"{cam}_{deg}.pcd"))


@lru_cache(maxsize=128)
def read_K(in_dir, cam, t="rgb", dtype=np.float):
    assert t in ("rgb", "ir")
    with h5py.File(join(in_dir, "calibration.h5"), "r") as f:
        return f[f'{cam}_{t}_K'][()].astype(dtype)


@lru_cache(maxsize=128)
def read_distortion(in_dir, cam, t="rgb", dtype=np.float):
    assert t in ("rgb", "ir")
    with h5py.File(join(in_dir, "calibration.h5"), "r") as f:
        return f[f'{cam}_{t}_d'][()].astype(dtype)


@lru_cache(maxsize=128)
def transformation_cam_to_cam(in_dir, cam_source, cam_target, type_source="rgb", type_target="rgb"):
    def type_to_str(t):
        assert t in ("rgb", "ir")
        if t == "rgb":
            return ""
        if t == "ir":
            return "ir_"

    with h5py.File(join(in_dir, "calibration.h5"), "r") as f:
        H_source_from_NP5 = f[f"H_{cam_source}_{type_to_str(type_source)}from_NP5"][()]
        H_target_from_NP5 = f[f"H_{cam_target}_{type_to_str(type_target)}from_NP5"][()]
    H_source_to_target = H_target_from_NP5 @ np.linalg.inv(H_source_from_NP5)

    return H_source_to_target


def transformation_cam_to_world(in_dir, cam, deg, t="rgb"):
    with h5py.File(join(in_dir, "poses", f"NP5_{deg}_pose.h5"), "r") as f:
        H_NP5_to_world = f['H_table_from_reference_camera'][()]

    H_cam_to_NP5 = transformation_cam_to_cam(in_dir, cam_source=cam, cam_target="NP5", type_source=t, type_target="rgb")
    H_cam_to_world = H_NP5_to_world @ H_cam_to_NP5

    return H_cam_to_world
