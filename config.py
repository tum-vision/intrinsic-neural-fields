import os
import torch
import torch.nn.functional as F
import yaml
from shutil import copyfile

from model import make_model
from renderer import Renderer
from mesh import load_first_k_eigenfunctions, load_mesh
from ray_dataloader import create_ray_dataloader


def _pretty_print_config(config, path):
    print("----------------------------------------------------------------")
    print(f"Loaded Config from {path}")
    print("================================================================")
    print(yaml.dump(config, default_flow_style=False))
    print("================================================================\n")


def _copy_config_file_into_out_dir(config, config_path):
    os.makedirs(config["training"]["out_dir"], exist_ok=True)
    copyfile(config_path, os.path.join(config["training"]["out_dir"], "config.yaml"))


def load_config_file(path, allow_checkpoint_loading=False):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    out_dir = config["training"]["out_dir"]
    if os.path.exists(out_dir) and not allow_checkpoint_loading:
        raise RuntimeError(f"out_dir '{out_dir}' exists. Exit to not overwrite old results.")

    _pretty_print_config(config, path)
    _copy_config_file_into_out_dir(config, path)
    return config


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_seed(config):
    return config.get("seed", 0)


def get_log_dir(config):
    if not os.path.exists(config["training"]["out_dir"]):
        os.makedirs(config["training"]["out_dir"])
    log_dir = os.path.join(config["training"]["out_dir"], "logs")
    return log_dir


def get_data(config, device, num_workers_per_data_loader=6):
    mesh = load_mesh(config["data"]["mesh_path"])
    data = {
        "train": create_ray_dataloader(config["data"]["preproc_data_path_train"],
                                       config["data"]["eigenfunctions_path"],
                                       config["model"].get("k"),
                                       config["model"].get("feature_strategy", "efuncs"),
                                       mesh,
                                       config["data"].get("rescale_strategy", "standard"),
                                       config["data"].get("embed_strategy"),
                                       config["data"].get("eigenvalues_path"),
                                       config["training"]["batch_size"],
                                       shuffle=True,
                                       drop_last=config["data"].get("train_drop_last", True),
                                       device=device),
        "val": create_ray_dataloader(config["data"]["preproc_data_path_eval"],
                                     config["data"]["eigenfunctions_path"],
                                     config["model"].get("k"),
                                     config["model"].get("feature_strategy", "efuncs"),
                                     mesh,
                                     config["data"].get("rescale_strategy", "standard"),
                                     config["data"].get("embed_strategy"),
                                     config["data"].get("eigenvalues_path"),
                                     config["training"]["batch_size"],
                                     shuffle=False,
                                     drop_last=False,
                                     device=device),
    }

    if hasattr(config["data"], "preproc_data_path_test"):
        data["test"] = create_ray_dataloader(config["data"]["preproc_data_path_test"],
                                      config["data"]["eigenfunctions_path"],
                                      config["model"].get("k"),
                                      config["model"].get("feature_strategy", "efuncs"),
                                      mesh,
                                      config["data"].get("rescale_strategy", "standard"),
                                      config["data"].get("embed_strategy"),
                                      config["data"].get("eigenvalues_path"),
                                      config["training"]["batch_size"],
                                      shuffle=False,
                                      drop_last=False,
                                      device=device)

    return data


def get_model_and_optim(config, mesh, device):
    model = make_model(config["model"], mesh=mesh)
    # Note: Moving the model to GPU should always be done BEFORE constructing the optimizer.
    # See https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    return model, optim


def get_loss_fn(config):
    loss_type = config["training"]["loss_type"]
    if loss_type == "L2":
        return F.mse_loss
    if loss_type == "L1":
        return F.l1_loss
    if loss_type == "cauchy":
        return lambda rgb_pred, rgb_gt: ((20 / 255) * (20 / 255) * torch.log(
            1 + (rgb_pred - rgb_gt) ** 2 / ((20 / 255) * (20 / 255)))).mean()
    raise RuntimeError(f"Unknown loss function: {loss_type}. Please use either 'L1', 'L2' or 'cauchy'")


def get_renderer(config, model, mesh, device):
    feature_strategy = config["model"].get("feature_strategy", "efuncs")
    if feature_strategy == "efuncs":
        E = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                        config["model"]["k"],
                                        rescale_strategy=config["data"].get("rescale_strategy", "standard"),
                                        embed_strategy=config["data"].get("embed_strategy"),
                                        eigenvalues_path=config["data"].get("eigenvalues_path"))
        return Renderer(model, mesh, eigenfunctions=E, H=config["data"]["img_height"],
                        W=config["data"]["img_width"], device=device)
    elif feature_strategy in ("ff", "rff", "xyz"):
        return Renderer(model, mesh, feature_strategy=feature_strategy,
                        H=config["data"]["img_height"], W=config["data"]["img_width"], device=device)
    else:
        raise ValueError(f"Unknown feature strategy: {feature_strategy}")
