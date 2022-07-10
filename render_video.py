def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import os, sys
if isnotebook() and sys.path[-1] != "../intrinsic-neural-fields":
    sys.path.append("../intrinsic-neural-fields")
from os.path import join, exists, basename, dirname
import argparse, shlex
import matplotlib.pyplot as plt
from fractions import Fraction

import numpy as np
import torch
from cameras import cam_resize, load_extr_and_intr_camera, cam_crop
from config import load_config
from renderer import make_renderer_with_trained_model

if isnotebook():
    print("LOADING tqdm.notebook")
    from tqdm.notebook import tqdm
else:
    print("LOADING tqdm for a python script")
    from tqdm import tqdm


resolutions = {
    "2160p": (3840, 2160),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
}

def parse_args(s=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--cameras_path", type=str)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--resolution", choices=tuple(resolutions.keys()))
    parser.add_argument("--turns", type=float, default=2)
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--fps", type=int, default=60)
    if s is None:
        # Called as script
        args = parser.parse_args()
    else:
        # Used in notebook
        args = parser.parse_args(shlex.split(s))
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)


    ### Make Renderer ###
    renderer = make_renderer_with_trained_model(config)
    def render_view(camCv2world_in, cam_intrinsic, pose_obj=np.eye(4)):
        pose_obj = torch.from_numpy(pose_obj).to(dtype=camCv2world_in.dtype, device=camCv2world_in.device)
        camCv2world = torch.eye(4, dtype=camCv2world_in.dtype, device=camCv2world_in.device)
        camCv2world[:3, :4] = camCv2world_in
        pose_render = pose_obj @ camCv2world
        renderer.set_height(cam_intrinsic["height"])
        renderer.set_width(cam_intrinsic["width"])
        return renderer.render(pose_render[:3], cam_intrinsic["K"])


    ### Change Intrinsics ###
    camCv2world, K = load_extr_and_intr_camera(args.cameras_path)
    cam_orig = {
        "K": K,
        "height": config["data"]["img_height"],
        "width": config["data"]["img_width"],
    }

    res = resolutions[args.resolution]
    factor_width = Fraction(res[0], cam_orig['width'])
    factor_height = Fraction(res[1], cam_orig['height'])
    factor = min(factor_height, factor_width)

    new_width = cam_orig['width']*factor
    new_height = cam_orig['height']*factor

    assert int(new_height) == new_height
    assert int(new_width) == new_width

    new_width, new_height = int(new_width), int(new_height)
    assert (res[0] - new_width) % 2 == 0
    assert (res[1] - new_height) % 2 == 0
    pad_width = (res[0] - new_width) //2
    pad_height = (res[1] - new_height) //2

    cam_resized = cam_resize(cam_orig, height=new_height, width=new_width)
    cam_render = cam_crop(cam_resized, height=res[1], width=res[0], col=-pad_width, row=-pad_height)

    # Assert that this is a "synthetic optimal" cam
    assert cam_render['K'][0,0] == cam_render['K'][1,1]
    assert cam_render['width']*0.5-0.5 == cam_render['K'][0,2]
    assert cam_render['height']*0.5-0.5 == cam_render['K'][1,2]

    ### Loop over rotations and render ###
    folder_name = basename(dirname(args.config_path))
    image_folder = join(args.out_dir, "images", folder_name+"_"+args.resolution)

    os.makedirs(image_folder, exist_ok=True)

    num_images = args.fps*args.duration
    assert num_images == int(num_images)
    num_images = int(num_images)
    angles = np.linspace(0, args.turns*360, num_images)

    for idx, a in enumerate(tqdm(angles)):
        st, ct = np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))
        rot = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]])

        pose_obj = np.eye(4)
        pose_obj[:3, :3] = rot

        view = render_view(camCv2world, cam_render, pose_obj=pose_obj)

        plt.imsave(join(image_folder, f"{idx:04d}.jpg"), view)


    ### Make video with ffmpeg ###
    video_name = join(args.out_dir, folder_name+"_"+args.resolution+".mp4")
    os.system(f"/usr/bin/ffmpeg -y -framerate {args.fps} -pattern_type glob -i '{image_folder}/*.jpg' -c:v libx264 -crf 17 -pix_fmt yuv420p {video_name}")
