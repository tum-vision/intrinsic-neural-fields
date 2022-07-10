import os
import numpy as np
import torch

from mesh import load_first_k_eigenfunctions, ray_tracing, get_ray_mesh_intersector, ray_tracing_xyz, load_mesh
from utils import to_device, load_trained_model, batchify_dict_data


def make_renderer_with_trained_model(config, device="cuda"):
    # Load mesh
    mesh = load_mesh(config["data"]["mesh_path"])

    # Load features
    feature_strategy = config["model"].get("feature_strategy", "efuncs")

    if feature_strategy == "efuncs":
        # Load eigenfunctions
        efuncs = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                             config["model"].get("k"),
                                             rescale_strategy=config["data"].get("rescale_strategy",
                                                                                 "standard"),
                                             embed_strategy=config["data"].get("embed_strategy"),
                                             eigenvalues_path=config["data"].get("eigenvalues_path"))
    else:
        efuncs = None

    # Load trained model
    weights_path = os.path.join(config["training"]["out_dir"], "model.pt")
    model = load_trained_model(config["model"], weights_path, device, mesh=mesh)

    return Renderer(model, mesh, eigenfunctions=efuncs, feature_strategy=feature_strategy,
                    device=device, H=config["data"]["img_height"], W=config["data"]["img_width"])


class Renderer:
    def __init__(self, model, mesh, eigenfunctions=None, feature_strategy="efuncs", background="white", device="cpu", *, H, W):
        self.model = model
        self.mesh = mesh
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        self.feature_strategy = feature_strategy
        
        if self.feature_strategy == "efuncs":
            self.features = eigenfunctions
        elif self.feature_strategy in ("ff", "rff", "xyz"):
            self.features = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown feature strategy: {self.feature_strategy}")
        
        self.H = H
        self.W = W
        self.background = background
        self.device = device

    def set_height(self, height):
        self.H = height

    def set_width(self, width):
        self.W = width
    
    def apply_mesh_transform(self, transform):
        self.mesh.apply_transform(transform)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)

    @torch.no_grad()
    def render(self, camCv2world, K, obj_mask_1d=None, eval_render=False, distortion_coeffs=None, distortion_type=None):
        assert obj_mask_1d is None or obj_mask_1d.size()[0] == self.H*self.W

        self.model.eval()

        if self.feature_strategy == "efuncs":
            eigenfunction_vector_values, hit_ray_idxs, unit_ray_dirs, face_idxs = ray_tracing(self.ray_mesh_intersector,
                                                                                              self.mesh,
                                                                                              self.features,
                                                                                              camCv2world,
                                                                                              K,
                                                                                              obj_mask_1d=obj_mask_1d,
                                                                                              H=self.H,
                                                                                              W=self.W,
                                                                                              batched=True,
                                                                                              distortion_coeffs=distortion_coeffs,
                                                                                              distortion_type=distortion_type)
            assert eigenfunction_vector_values.dtype == torch.float32
            data = {
                "eigenfunctions": eigenfunction_vector_values,
                "unit_ray_dirs": unit_ray_dirs,
                "hit_face_idxs": face_idxs,
            }
            num_rays = eigenfunction_vector_values.shape[0]
        elif self.feature_strategy in ("ff", "rff", "xyz"):
            hit_points_xyz, hit_ray_idxs, unit_ray_dirs, face_idxs = ray_tracing_xyz(self.ray_mesh_intersector,
                                                                                     self.mesh,
                                                                                     self.features,
                                                                                     camCv2world,
                                                                                     K,
                                                                                     obj_mask_1d=obj_mask_1d,
                                                                                     H=self.H,
                                                                                     W=self.W,
                                                                                     batched=True,
                                                                                     distortion_coeffs=distortion_coeffs,
                                                                                     distortion_type=distortion_type)
            data = {
                "xyz": hit_points_xyz,
                "unit_ray_dirs": unit_ray_dirs,
                "hit_face_idxs": face_idxs
            }
            num_rays = hit_points_xyz.shape[0]
        else:
            raise ValueError(f"Unknown feature strategy: {self.feature_strategy}")

        assert num_rays > 0

        # Inference in batches to support rendering large views
        total_pred_rgbs = []
        batch_size = 1 << 15
        for batch in batchify_dict_data(data, num_rays, batch_size):
            batch = to_device(batch, device=self.device)
            pred_rgbs = self.model(batch).cpu()
            total_pred_rgbs.append(pred_rgbs)
        pred_rgbs = torch.concat(total_pred_rgbs, dim=0)

        # We now need to bring the predicted RGB colors into the correct ordering again
        # since the ray-mesh intersection does not preserve ordering
        assert obj_mask_1d is None or obj_mask_1d.dtype == torch.bool
        N = self.H * self.W if obj_mask_1d is None else obj_mask_1d.sum()
        if self.background == "white":
            img = torch.ones((N, 3), device="cpu", dtype=torch.float32)
        else:
            assert self.background == "black"
            img = torch.zeros((N, 3), device="cpu", dtype=torch.float32)
        img[hit_ray_idxs] = pred_rgbs
        
        # If we only kept the object, then img does not have the correct resolution yet.
        # Therefore, we must upscale it one more time taking the object mask into account.
        if obj_mask_1d is not None:
            M = self.H * self.W
            if self.background == "white":
                img_unmasked = torch.ones((M, 3), device="cpu", dtype=torch.float32)
            else:
                assert self.background == "black"
                img_unmasked = torch.zeros((M, 3), device="cpu", dtype=torch.float32)
            img_unmasked[obj_mask_1d] = img
            img = img_unmasked

        if eval_render:
            return img.reshape(self.H, self.W, 3), hit_ray_idxs
        return img.reshape(self.H, self.W, 3).numpy()
