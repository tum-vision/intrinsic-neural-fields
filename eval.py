import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import random
# https://github.com/richzhang/PerceptualSimilarity#a-basic-usage
import lpips

from utils import load_trained_model
from dataset import MeshViewsDataset, MeshroomRadialK3Dataset
from config import load_config, get_seed
from mesh import load_mesh, get_ray_mesh_intersector, load_first_k_eigenfunctions
from evaluation_metrics import psnr, dssim
from bake_texture_field import bake_texture
from renderer import Renderer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("--uv_mesh_path", type=str, default=None)
    parser.add_argument("--background", nargs='?', type=str, default="white")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.uv_mesh_path is not None:
        print("Baking texture into UV-map...")
        bake_texture(args.output_path, args.uv_mesh_path, args.config_path)
        print("Done.")

    config = load_config(args.config_path)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    seed = get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    H, W = config["data"]["img_height"], config["data"]["img_width"]

    dataset_type = config["data"].get("type")
    if dataset_type is None:
        dataset = MeshViewsDataset(args.dataset_path,
                                   args.split,
                                   H=H,
                                   W=W,
                                   background=args.background)
    elif dataset_type == "meshroom_radial_k3":
        dataset = MeshroomRadialK3Dataset(args.dataset_path, 
                                          args.split, 
                                          H=H, 
                                          W=W)
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             shuffle=False,
                                             drop_last=False)

    # Load mesh
    mesh = load_mesh(config["data"]["mesh_path"])

    # Load features
    feature_strategy = config["model"].get("feature_strategy", "efuncs")

    if feature_strategy == "efuncs":
        # Load eigenfunctions
        features = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                               config["model"].get("k"),
                                               rescale_strategy=config["data"].get("rescale_strategy",
                                                                                   "standard"),
                                               embed_strategy=config["data"].get("embed_strategy"),
                                               eigenvalues_path=config["data"].get("eigenvalues_path"))
    elif feature_strategy in ("xyz", "ff", "rff"):
        features = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown feature strategy: {feature_strategy}")

    # Ray-mesh intersector
    ray_mesh_intersector = get_ray_mesh_intersector(mesh)

    # Load trained model
    weights_path = os.path.join(config["training"]["out_dir"], "model.pt")

    model = load_trained_model(config["model"],
                               weights_path,
                               device,
                               mesh=mesh)
    model = model.eval()

    # Process each view
    eval_metrics_results = {}

    os.makedirs(args.output_path, exist_ok=True)

    lpips_fn = lpips.LPIPS(net='alex')

    total_psnr = 0
    total_dssim = 0
    total_lpips = 0
    total = 0

    if feature_strategy == "efuncs":
        renderer = Renderer(model, mesh, eigenfunctions=features,
                            feature_strategy=feature_strategy, H=H, W=W, device=device)
    else:
        assert feature_strategy in ("xyz", "ff", "rff")
        renderer = Renderer(model, mesh, feature_strategy=feature_strategy, H=H, W=W, device=device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            view_id = f"{i:03d}"

            camCv2world = batch["camCv2world"]
            K = batch["K"]
            real_img = batch["img"]
            obj_mask_1d = batch["obj_mask_1d"]
            distortion_params = batch.get("distortion_params")
            distortion_type = batch.get("distortion_type")

            fake_img, hit_ray_idxs = renderer.render(camCv2world, 
                                                     K, 
                                                     eval_render=True, 
                                                     distortion_coeffs=distortion_params, 
                                                     distortion_type=distortion_type)

            # Use hit ray mask & (logical) obj_mask instead of object mask due to some minor errors within the
            # ray-mesh intersection.
            # The hit_ray_mask tells us which pixels hit the mesh.
            hit_ray_mask = torch.zeros(H * W, dtype=torch.bool)
            hit_ray_mask[hit_ray_idxs] = True
            obj_mask_1d = torch.logical_and(hit_ray_mask, obj_mask_1d)

            # Store an unmasked version of the rendering
            fake_img_raw = fake_img.clone().numpy()

            # Mask out background of both images
            bg_mask_1d = obj_mask_1d == False

            fake_img = fake_img.reshape(-1, 3)
            fake_img[bg_mask_1d] = 1.
            fake_img = fake_img.reshape(H, W, 3)

            real_img = real_img.reshape(-1, 3)
            real_img[bg_mask_1d] = 1.
            real_img = real_img.reshape(H, W, 3)

            lpips_input_real = real_img.permute(2, 0, 1).unsqueeze(0)
            lpips_input_fake = fake_img.permute(2, 0, 1).unsqueeze(0)

            fake_img = fake_img.numpy()
            real_img = real_img.numpy()

            # Evaluation Metrics
            metrics = {
                "psnr": psnr(fake_img, real_img, obj_mask_1d),
                "dssim_rescaled": dssim(fake_img, real_img) * 100,
                "lpips_rescaled": lpips_fn(lpips_input_fake, lpips_input_real).item() * 100,
            }

            total_psnr += metrics["psnr"]
            total_dssim += metrics["dssim_rescaled"]
            total_lpips += metrics["lpips_rescaled"]
            total += 1

            # Store rendered view and evaluation metrics
            eval_metrics_results[view_id] = metrics
            plt.imsave(os.path.join(args.output_path, f"{view_id}_fake_raw.png"), fake_img_raw)
            plt.imsave(os.path.join(args.output_path, f"{view_id}_fake.png"), fake_img)
            plt.imsave(os.path.join(args.output_path, f"{view_id}_real.png"), real_img)

    # Store the metrics results.
    with open(os.path.join(args.output_path, "evaluation_metrics.pkl"), "wb") as f:
        pickle.dump(eval_metrics_results, f)

    print(f"PSNR: {total_psnr / total}, DSSIM: {total_dssim / total}, LPIPS: {total_lpips / total}")


if __name__ == "__main__":
    main()
