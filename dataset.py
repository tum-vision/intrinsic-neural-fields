import numpy as np
import torch
import os
import imageio
import json

from cameras import DistortionTypes
from mesh import get_k_eigenfunc_vec_vals, load_first_k_eigenfunctions
from utils import load_obj_mask_as_tensor, load_cameras


def load_preprocessed_data(preproc_data_path):
    data = {}
    
    vertex_idxs_of_hit_faces = np.load(os.path.join(preproc_data_path, "vids_of_hit_faces.npy"))
    data["vertex_idxs_of_hit_faces"] = torch.from_numpy(vertex_idxs_of_hit_faces).to(dtype=torch.int64)

    barycentric_coords = np.load(os.path.join(preproc_data_path, "barycentric_coords.npy"))
    data["barycentric_coords"] = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)

    expected_rgbs = np.load(os.path.join(preproc_data_path, "expected_rgbs.npy"))
    data["expected_rgbs"] = torch.from_numpy(expected_rgbs).to(dtype=torch.float32)
    
    unit_ray_dirs_path = os.path.join(preproc_data_path, "unit_ray_dirs.npy")
    face_idxs_path = os.path.join(preproc_data_path, "face_idxs.npy")
    if os.path.exists(unit_ray_dirs_path) and os.path.exists(face_idxs_path):
        unit_ray_dirs = np.load(unit_ray_dirs_path)
        data["unit_ray_dirs"] = torch.from_numpy(unit_ray_dirs).to(dtype=torch.float32)

        face_idxs = np.load(face_idxs_path)
        data["face_idxs"] = torch.from_numpy(face_idxs).to(dtype=torch.int64)
    
    return data


class MeshViewsPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 preproc_data_path, 
                 eigenfunctions_path, 
                 k,
                 feature_strategy="efuncs",
                 mesh=None,
                 rescale_strategy="standard",
                 eigenvalues_path=None, 
                 embed_strategy=None,
                 transforms=None):
        assert os.path.exists(preproc_data_path)
        self.feature_strategy = feature_strategy
        if self.feature_strategy == "efuncs":
            self.k = k
            self.E = load_first_k_eigenfunctions(eigenfunctions_path,
                                                 self.k,
                                                 rescale_strategy=rescale_strategy,
                                                 embed_strategy=embed_strategy,
                                                 eigenvalues_path=eigenvalues_path)
        elif self.feature_strategy in ("ff", "rff", "xyz"):
            assert mesh is not None
            self.vertices = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown input feature strategy: {self.feature_strategy}")

        data = load_preprocessed_data(preproc_data_path)
        self.vertex_idxs_of_hit_faces = data["vertex_idxs_of_hit_faces"]
        self.barycentric_coords = data["barycentric_coords"]
        self.expected_rgbs = data["expected_rgbs"]
        
        self.unit_ray_dirs = data.get("unit_ray_dirs")
        self.face_idxs = data.get("face_idxs")

        self.transforms = transforms
        
    def get_eigenfunctions(self):
        return self.E

    def __len__(self):
        return len(self.expected_rgbs)
    
    def __getitem__(self, idx):
        item = {}
        
        assert idx < len(self.expected_rgbs)
        vertex_idxs_of_hit_faces = self.vertex_idxs_of_hit_faces[idx]  # 3
        barycentric_coords = self.barycentric_coords[idx]  # 3

        if self.feature_strategy == "efuncs":
            eigenfuncs = get_k_eigenfunc_vec_vals(self.E, vertex_idxs_of_hit_faces.unsqueeze(0), barycentric_coords.unsqueeze(0))
            assert eigenfuncs.dtype == torch.float32
            item["eigenfunctions"] = eigenfuncs.squeeze(0)
        elif self.feature_strategy in ("ff", "rff", "xyz"):
            item["xyz"] = self.vertices[vertex_idxs_of_hit_faces].T @ barycentric_coords
        else:
            raise ValueError(f"Unknown input feature strategy: {self.feature_strategy}")

        expected_rgbs = self.expected_rgbs[idx]
        assert expected_rgbs.dtype == torch.float32
        item["expected_rgbs"] = expected_rgbs
        
        if self.unit_ray_dirs is not None:
            assert self.face_idxs is not None
            item["unit_ray_dirs"] = self.unit_ray_dirs[idx]
            item["hit_face_idxs"] = self.face_idxs[idx]
            
        if self.transforms is not None:
            return self.transforms(item)

        return item


class MeshViewsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, H=512, W=512, background="white"):
        self.dataset_path = dataset_path
        self.H = H
        self.W = W
        self.background = background
        with open(os.path.join(self.dataset_path, f"{split}.lst"), "r") as file_handle:
            self.mesh_views_list = [line[:-1] if line.endswith('\n') else line for line in file_handle.readlines()]
            
    def __len__(self):
        return len(self.mesh_views_list)
    
    def __getitem__(self, idx):
        assert idx < len(self.mesh_views_list)
        
        mesh_view_path = os.path.join(self.dataset_path, self.mesh_views_list[idx])

        # Load cameras
        camCv2world, K = load_cameras(mesh_view_path)

        # Load object mask
        obj_mask = load_obj_mask_as_tensor(mesh_view_path)
        bg_mask_1d = (obj_mask == False).reshape(-1)
        obj_mask_1d = obj_mask.reshape(-1)

        # Load view image
        img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))
        img = torch.from_numpy(img).to(dtype=torch.float32)
        img /= 255.
        img = img.reshape(-1, 3)
        # Ensure that background is correct and everything besides the object is set to the background color.
        if self.background == "white":
            img[bg_mask_1d] = 1.0
        else:
            assert False, "Currently only white background is supported"
        img = img.reshape(self.H, self.W, 3)

        return {
            "camCv2world": camCv2world,
            "K": K,
            "img": img,
            "obj_mask_1d": obj_mask_1d
        }


#=== Meshroom Radial K3


def load_meshroom_metadata(dataset_path, split):
    with open(os.path.join(dataset_path, f"{split}_data.json"), "r") as file_handle:
        metadata = json.load(file_handle)
    return metadata


class MeshroomRadialK3Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, *, H, W):
        self.dataset_path = dataset_path
        self.H = H
        self.W = W
        self.metadata = load_meshroom_metadata(dataset_path, split)
        
        self.K = torch.from_numpy(np.array(self.metadata["K"]).astype(np.float32))
        self.distortion_params = list(map(float, self.metadata["distortion_params"]))

    def __len__(self):
        return len(self.metadata["views"])
    
    def __getitem__(self, idx):
        assert idx < len(self.metadata["views"])
        
        cur_view = self.metadata["views"][idx]
        
        # Load view image
        img = imageio.imread(os.path.join(self.dataset_path, cur_view["view_file"])) / 255.
        img = torch.from_numpy(img).to(dtype=torch.float32)

        # Get masks
        obj_mask = np.load(os.path.join(self.dataset_path, cur_view["obj_mask_file"]))
        bg_mask = obj_mask == False
        
        # Mask out background of the image
        img[bg_mask] = 1.

        cam2world = torch.from_numpy(np.array(cur_view["cam2world"]).astype(np.float32))
        cam2world = cam2world[:3]  # 3x4

        return {
            "camCv2world": cam2world,
            "K": self.K,
            "distortion_params": self.distortion_params,
            "distortion_type": DistortionTypes.MESHROOM_RADIAL_K3,
            "img": img,
            "obj_mask_1d": obj_mask.reshape(-1)
        }
