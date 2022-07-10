import numpy as np
import torch
import sys
from torchinfo import summary
import os
import imageio

from model import make_model

# Make sure loading .exr works for imageio
try:
    imageio.plugins.freeimage.download()
except FileExistsError:
    # Ignore
    pass


def tensor_mem_size_in_bytes(x):
    return sys.getsizeof(x.storage())


def load_trained_model(model_config, weights_path, device, mesh=None):
    model = make_model(model_config, mesh=mesh)
    data = torch.load(weights_path)
    if "model_state_dict" in data:
        model.load_state_dict(data["model_state_dict"])
    else:
        model.load_state_dict(data)
    return model.to(device)


def load_cameras(view_path):
    cameras = np.load(os.path.join(view_path, "depth", "cameras.npz"))
    camCv2world = torch.from_numpy(cameras["world_mat_0"]).to(dtype=torch.float32)
    K = torch.from_numpy(cameras["camera_mat_0"]).to(dtype=torch.float32)
    return camCv2world, K


def model_summary(model, data):
    data_batch = next(iter(data["train"]))
    summary(model, input_data=[data_batch])


def load_obj_mask_as_tensor(view_path):
    if view_path.endswith(".npy"):
        return np.load(view_path)

    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    if os.path.exists(depth_path):
        depth_map = imageio.imread(depth_path)[..., 0]

        mask_value = 1.e+10
        obj_mask = depth_map != mask_value
    else:
        mask_path = os.path.join(view_path, "depth", "mask.png")
        assert os.path.exists(mask_path), "Must have depth or mask"
        mask = imageio.imread(mask_path)
        obj_mask = mask != 0  # 0 is invalid

    obj_mask = torch.from_numpy(obj_mask)
    return obj_mask


def load_depth_as_numpy(view_path):
    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    assert os.path.exists(depth_path)
    depth_map = imageio.imread(depth_path)[..., 0]

    return depth_map


def batchify_dict_data(data_dict, input_total_size, batch_size):
    idxs = np.arange(0, input_total_size)
    batch_idxs = np.split(idxs, np.arange(batch_size, input_total_size, batch_size), axis=0)

    batches = []
    for cur_idxs in batch_idxs:
        data = {}
        for key in data_dict.keys():
            data[key] = data_dict[key][cur_idxs]
        batches.append(data)

    return batches


##########################################################################################
# The following is taken from:
# https://github.com/tum-vision/tandem/blob/master/cva_mvsnet/utils.py
##########################################################################################


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, **kwargs)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def to_device(x, *, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, str):
        return x
    else:
        raise NotImplementedError(f"Invalid type for to_device: {type(x)}")

##########################################################################################
