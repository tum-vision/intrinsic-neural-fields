"""
Partly taken from
https://github.com/tum-vision/tandem/blob/master/cva_mvsnet/models/datasets.py#L144
"""
import numpy as np
import torch


def _fx(cam: dict) -> float:
    return cam['K'][0, 0]


def _cx(cam: dict) -> float:
    return cam['K'][0, 2]


def _fy(cam: dict) -> float:
    return cam['K'][1, 1]


def _cy(cam: dict) -> float:
    return cam['K'][1, 2]


def _height(cam: dict) -> int:
    return cam['height']


def _width(cam: dict) -> int:
    return cam['width']


def _is_torch(cam: dict) -> bool:
    return torch.is_tensor(cam['K'])


def cam_resize(cam: dict,
               height: int,
               width: int):
    """
    Convert to new camera intrinsics for resize of image from original camera.
    :param cam:
        camera intrinsics
    :param height:
        height of resized frame
    :param width:
        width of resized frame
    :return:
        camera intrinsics for resized frame
    """
    center_x = 0.5 * float(_width(cam) - 1)
    center_y = 0.5 * float(_height(cam) - 1)

    orig_cx_diff = _cx(cam) - center_x
    orig_cy_diff = _cy(cam) - center_y

    scaled_center_x = 0.5 * float(width - 1)
    scaled_center_y = 0.5 * float(height - 1)

    scale_x = float(width) / float(_width(cam))
    scale_y = float(height) / float(_height(cam))

    fx = scale_x * _fx(cam)
    fy = scale_y * _fy(cam)
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    if _is_torch(cam):
        return {
            "K": torch.tensor([[fx, 0, cx, 0],
                               [0, fy, cy, 0],
                               [0, 0, 1, 0]], dtype=torch.float32),
            "height": height,
            "width": width,
        }
    else:
        return {
            "K": np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0]]),
            "height": height,
            "width": width,
        }


def cam_crop(cam: dict,
             height: int,
             width: int,
             col: int,
             row: int):
    fx = _fx(cam)
    fy = _fy(cam)
    cx = _cx(cam) - col
    cy = _cy(cam) - row

    if _is_torch(cam):
        return {
            "K": torch.tensor([[fx, 0, cx, 0],
                               [0, fy, cy, 0],
                               [0, 0, 1, 0]], dtype=torch.float32),
            "height": height,
            "width": width,
        }
    else:
        return {
            "K": np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0]]),
            "height": height,
            "width": width,
        }


def load_extr_and_intr_camera(camera_path):
    cameras = np.load(camera_path)
    camCv2world = torch.from_numpy(cameras["world_mat_0"]).to(dtype=torch.float32)
    K = torch.from_numpy(cameras["camera_mat_0"]).to(dtype=torch.float32)
    return camCv2world, K


def _principal_point(K):
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/Pinhole.hpp#L74
    return K[:2, 2]


def _focal(K):
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/Pinhole.hpp#L73
    return K[0,0]


# Solve a single variable nonlinear equation
# Find p' such that disto_func(p') = r2 approximately holds
def _bisection_radius_solve(r2, disto_func):
    eps = 1e-8

    # Guess plausible upper and lower bound
    lb, ub = r2, r2
    while disto_func(lb) > r2:
        lb /= 1.05
    while disto_func(ub) < r2:
        ub *= 1.05

    # Bisection until accuracy is reached
    while eps < (ub - lb):
        m = (lb + ub) / 2
        if disto_func(m) > r2:
            ub = m
        else:
            lb = m
    
    return (lb + ub) / 2


def _remove_distortion(p, disto_func):
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/PinholeRadial.hpp#L167
    r2 = p[:,0]*p[:,0] + p[:,1]*p[:,1]
    for i in range(p.shape[0]):
        if r2 == 0:
            radius = 1
        else:
            radius = np.sqrt(_bisection_radius_solve(r2[i], disto_func) / r2)
        p[i] *= radius
    return p


# Vectorized version of _bisection_radius_solve
def _bisection_radius_solve_v2(r2, disto_func, radius_one_mask):
    eps = 1e-8
    
    f = lambda ps: disto_func(ps) - r2

    # Guess plausible upper and lower bound
    lb, ub = np.array(r2), np.array(r2)
    while True:
        cond = f(lb) > 0
        cond[radius_one_mask] = False
        if not np.any(cond):
            break
        lb[cond] /= 1.05

    while True:
        cond = f(ub) < 0
        cond[radius_one_mask] = False
        if not np.any(cond):
            break
        ub[cond] *= 1.05

    # Bisection until accuracy is reached for every entry
    while True:
        cond = eps < (ub - lb)
        cond[radius_one_mask] = False
        if not np.any(cond):
            break
        
        m = (lb + ub) / 2
        cond2 = f(m) > 0
        
        mask_ub = np.logical_and(cond, cond2)
        ub[mask_ub] = m[mask_ub]
        mask_lb = np.logical_and(cond, cond2 == False)
        lb[mask_lb] = m[mask_lb]

    return (lb + ub) / 2
    

# Vectorized version of _remove_distortion
def _remove_distortion_v2(p, disto_func):
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/PinholeRadial.hpp#L167
    r2 = p[:,0]*p[:,0] + p[:,1]*p[:,1]
    radius_one_mask = r2 == 0
    
    radius = np.sqrt(_bisection_radius_solve_v2(r2, disto_func, radius_one_mask) / r2)
    radius[radius_one_mask] = 1
    
    return p * radius[..., None]


def undistort_pixels_meshroom_radial_k3(p_2d, K, distortion):
    # The pixels are distorted.
    # Undistortion => cam2ima( remove_disto(ima2cam(p)) )
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/PinholeRadial.hpp#L179
    
    # See https://github.com/alicevision/AliceVision/blob/d7a737f5d8b4ae32ca5f389c8266c49c4e733894/src/aliceVision/camera/Pinhole.hpp#L84 
    # cam2ima = focal() * p + principal_point()
    # ima2cam = ( p -  principal_point() ) / focal()
    focal = _focal(K)
    principal_point = _principal_point(K)
    # Transform a point from the camera plane to the image plane
    cam2ima = lambda p: focal * p + principal_point
    # Transform a point from the image plane to the camera plane
    ima2cam = lambda p: (p - principal_point) / focal

    k1 = distortion[0]
    k2 = distortion[1]
    k3 = distortion[2]
    square = lambda x: x*x
    disto_func = lambda x: x * square(1 + x * (k1 + x * (k2 + x * k3)))  # x == r2

    return cam2ima(_remove_distortion_v2(ima2cam(p_2d), disto_func))


# Supported distortion types
class DistortionTypes:
    MESHROOM_RADIAL_K3 = "meshroom_radial_k3"
