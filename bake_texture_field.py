import numpy as np
import torch
import trimesh
import os
from scipy.spatial import cKDTree
import shutil
import cv2
import warnings
from scipy.signal import convolve2d
import argparse
import matplotlib.pyplot as plt

from config import load_config
from mesh import load_first_k_eigenfunctions, load_mesh
from utils import load_trained_model, to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv_mesh", type=str, help="Path to the mesh that has the uv-map. Must be exported from meshlab so that the quad mesh is converted to a tri mesh (Include tex-coordinates and turn off polygonal during export). Will be loaded with trimesh.load_mesh.")
    parser.add_argument("--config_path", type=str, help="Path to config file for a trained model.", required=True)
    parser.add_argument("--out_dir", type=str, help="Path to out folder. May not exists to avoid accidental overwrite.",
                        required=True)
    args = parser.parse_args()
    return args


def area_tri(a, b, c):
    assert a.shape == b.shape == c.shape
    assert a.ndim == 2 and a.shape[-1] == 2  # (T, 2)

    v0 = a - c
    v1 = b - c
    return 0.5 * (v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0])


def point_in_tri(p, a, b, c):
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert a.shape == b.shape == c.shape
    assert a.ndim == 2 and a.shape[-1] == 2  # (T, 2)

    def sign(p1, p2, p3):
        assert p1.shape == p2.shape == p3.shape
        assert p1.shape[-1] == 2
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - (p2[..., 0] - p3[..., 0]) * (
                p1[..., 1] - p3[..., 1])

    N, T = p.shape[0], a.shape[0]

    # all to N, T, 2
    p = np.tile(p[:, None], [1, T, 1])
    a = np.tile(a[None], [N, 1, 1])
    b = np.tile(b[None], [N, 1, 1])
    c = np.tile(c[None], [N, 1, 1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 <= 0) | (d2 <= 0) | (d3 <= 0)
    has_pos = (d1 >= 0) | (d2 >= 0) | (d3 >= 0)

    return np.logical_not(has_neg & has_pos)


def point_in_tri_matched(p, a, b, c):
    # p (N, 2)
    # a (N, T, 2)
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert a.shape == b.shape == c.shape
    assert a.ndim == 3 and a.shape[0] == p.shape[0] and a.shape[-1] == 2  # (N, T, 2)

    def sign(p1, p2, p3):
        assert p1.shape == p2.shape == p3.shape
        assert p1.shape[-1] == 2
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - (p2[..., 0] - p3[..., 0]) * (
                p1[..., 1] - p3[..., 1])

    N, T, _ = a.shape

    # all to N, T, 2
    p = np.tile(p[:, None], [1, T, 1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    #     has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    #     has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    has_neg = (d1 <= 0) | (d2 <= 0) | (d3 <= 0)
    has_pos = (d1 >= 0) | (d2 >= 0) | (d3 >= 0)

    return np.logical_not(has_neg & has_pos)


def clean_tris(fun, min_area=1e-4):
    assert min_area > 0

    def cleaned_fun(p, a, b, c, **kwargs):
        areas = np.abs(area_tri(a=a, b=b, c=c))
        assert areas.ndim == 1
        idx_good_tri = np.where(areas >= min_area)[0]
        assert idx_good_tri.ndim == 1 and idx_good_tri.size <= areas.size

        idx_inter = fun(p=p, a=a[idx_good_tri], b=b[idx_good_tri], c=c[idx_good_tri])

        idx_out = np.copy(idx_inter)
        idx_out[idx_inter >= 0] = idx_good_tri[idx_inter[idx_inter >= 0]]
        return idx_out

    return cleaned_fun


@clean_tris
def get_tris_naive(p, a, b, c):
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert a.shape == b.shape == c.shape
    assert a.ndim == 2 and a.shape[-1] == 2  # (T, 2)

    mask = point_in_tri(p=p, a=a, b=b, c=c)  # (T, N)

    num_tris = np.sum(mask, -1)  # (T)
    assert np.all((num_tris == 0) | (num_tris == 1))

    idx = np.argmax(mask, axis=-1)  # (N)
    idx_mask_vals = mask[(np.arange(len(mask)), idx)]
    assert np.array_equal(idx_mask_vals, np.any(mask, axis=1))
    idx[np.logical_not(idx_mask_vals)] = -1

    return idx  # (N)


@clean_tris
def get_tris_fast(p, a, b, c, num_neigh=10):
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert a.shape == b.shape == c.shape
    assert a.ndim == 2 and a.shape[-1] == 2  # (T, 2)

    centroids = (a + b + c) / 3
    tree = cKDTree(centroids)

    dists, idx_partial = tree.query(p, k=num_neigh)

    ai = a[idx_partial]
    bi = b[idx_partial]
    ci = c[idx_partial]

    mask_matched = point_in_tri_matched(p=p, a=ai, b=bi, c=ci)  # (N, num_neighs)
    num_tris = np.sum(mask_matched, -1)  # (T)
    if num_tris.max() > 1:
        warnings.warn(
            f"A point was matched to {num_tris.max()} triangles. Overall {np.sum(num_tris > 1)} points were matched with more than one triangle. Selection will be random.")

    idx_matched = np.argmax(mask_matched, axis=-1)  # (N)
    idx_matched_mask_vals = mask_matched[(np.arange(len(mask_matched)), idx_matched)]
    assert np.array_equal(idx_matched_mask_vals, np.any(mask_matched, axis=1))

    idx = idx_partial[(np.arange(len(idx_matched)), idx_matched)]
    idx[np.logical_not(idx_matched_mask_vals)] = -1

    return idx


def bary(p, a, b, c, abs_tol=1e-10):
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert a.shape == b.shape == c.shape
    assert a.ndim == 2 and a.shape[-1] == 2  # (T, 2)

    def dot(x, y):
        assert x.shape == y.shape
        assert x.shape[-1] == 2
        return np.sum(x * y, -1)

    N = p.shape[0]

    v0 = np.tile((b - a)[None], [N, 1, 1])  # N, T, 2
    v1 = np.tile((c - a)[None], [N, 1, 1])  # N, T, 2
    v2 = p[:, None, :] - a[None, :, :]  # N, T, 2
    assert v0.shape == v1.shape == v2.shape

    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)

    denom = np.maximum(d00 * d11 - d01 * d01, abs_tol)
    print(np.amin(denom))
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def bary_matched(p, a, b, c, abs_tol=0):
    assert p.ndim == 2 and p.shape[-1] == 2  # (N, 2)
    assert p.shape == a.shape == b.shape == c.shape

    def dot(x, y):
        assert x.shape == y.shape
        assert x.shape[-1] == 2
        return np.sum(x * y, -1)

    N = p.shape[0]

    v0 = b - a  # N, 2
    v1 = c - a  # N, 2
    v2 = p - a  # N, 2
    assert v0.shape == v1.shape == v2.shape

    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)

    denom = np.maximum(d00 * d11 - d01 * d01, abs_tol)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    assert u.ndim == 1 and u.shape[0] == N
    assert u.shape == v.shape == w.shape
    rec = u[:, None] * a + v[:, None] * b + w[:, None] * c
    assert np.allclose(rec, p)

    return u, v, w


def xyz_from_faces_and_bary(vert, faces, bari):
    assert vert.ndim == 2 and faces.ndim == 2 and bari.ndim == 2
    assert vert.shape[-1] == 3 and faces.shape[-1] == 3
    assert bari.shape == faces.shape

    assert np.allclose(np.sum(bari, -1), 1)
    assert np.all(bari >= 0)

    a, b, c = vert[faces[:, 0]], vert[faces[:, 1]], vert[faces[:, 2]]  # (N, 3)
    u, v, w = bari[:, 0], bari[:, 1], bari[:, 2]

    return a * u[..., None] + b * v[..., None] + c * w[..., None]


def uv_fill_holes(CC):
    assert CC.ndim == 3 and CC.shape[-1] == 3

    kernel = np.array([1., 4, 6, 4, 1])
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / np.sum(kernel)

    CCf = np.stack([convolve2d(CC[..., i], kernel, mode="same", boundary="fill", fillvalue=0.0) for i in range(3)], -1)
    CC_out = np.copy(CC)

    mask = np.any(CC != 0, axis=-1)
    Wf = convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=0.0)
    mask_fill = np.logical_not(mask) & (Wf > 0)

    CC_out[mask_fill] = CCf[mask_fill] / Wf[mask_fill, None]

    assert np.all(CC[CC > 0] == CC_out[CC > 0])

    return CC_out


@torch.no_grad()
def pred_rgbs(mesh, faces_index_efs, barycentric_coords, config):
    feature_strategy = config["model"].get("feature_strategy", "efuncs")

    view_dependence_config = config["model"].get("view_dependence", None)
    if view_dependence_config is not None:
        raise NotImplementedError("Currently view dependence is not supported.")

    # Load a trained model
    weights_path = os.path.join(config["training"]["out_dir"], "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(config["model"], weights_path, device, mesh=mesh)
    model = model.eval()

    if feature_strategy == "efuncs":
        # Load eigenfunctions
        features = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                               config["model"].get("k"),
                                               rescale_strategy=config["data"].get("rescale_strategy", "standard"),
                                               embed_strategy=config["data"].get("embed_strategy"),
                                               eigenvalues_path=config["data"].get("eigenvalues_path"))
        features = features.numpy()
        key = "eigenfunctions"
    elif feature_strategy in ("xyz", "ff", "rff"):
        features = np.array(mesh.vertices)
        key = "xyz"
    else:
        raise ValueError(f"Unknown feature strategy: {feature_strategy}")

    out = []
    num_hit_points = faces_index_efs.shape[0]

    batch_size = 1 << 15
    for batch_id in range((num_hit_points + batch_size - 1)//batch_size):
        start = batch_id * batch_size
        end = min(start + batch_size, num_hit_points)
        length = end - start

        cur_faces_index_efs = faces_index_efs[start:end]
        cur_barycentric_coords = barycentric_coords[start:end]

        # Calculate features
        selected_features = features[cur_faces_index_efs.reshape(-1)].reshape(length, 3, -1)  # N x 3 x k
        selected_features = np.einsum('bij,bi->bj', selected_features, cur_barycentric_coords)
        selected_features = torch.from_numpy(selected_features.astype(np.float32))

        batch = to_device({
            key: selected_features
        }, device=device)

        # Inference
        out.append(model(batch).detach().cpu().numpy())

    return np.concatenate(out, axis=0)


def get_diffuse_color_map_file_name(uv_mesh_path):
    mtl_file_path = uv_mesh_path + ".mtl"
    with open(mtl_file_path, "r") as file_handle:
        lines = file_handle.readlines()
    target_line = [line for line in lines if line.startswith("map_Kd")]
    if len(target_line) != 1:
        raise ValueError(f".mtl File {mtl_file_path} is missing 'map_Kd'")
    file_name = target_line[0].split()[1]
    file_name = file_name[:-1] if file_name.endswith('\n') else file_name
    return os.path.basename(file_name)


def bake_texture(out_dir, uv_mesh_path, config_path):
    # Out dir
    assert not os.path.exists(out_dir)
    os.makedirs(out_dir)

    config = load_config(config_path)

    # Mesh Loading
    m = trimesh.load_mesh(uv_mesh_path)
    m_efs = load_mesh(config["data"]["mesh_path"])

    assert m_efs.faces.shape == m.faces.shape
    assert len(m.vertices) >= len(m_efs.vertices)
    assert len(np.unique(m.vertices, axis=0)) == len(m_efs.vertices)
    print(f"Created {len(m.vertices) - len(m_efs.vertices)} additional vertices for texture stuff")

    # Correspondences
    tree_efs = cKDTree(m_efs.vertices)
    _, idx_uv_to_ef = tree_efs.query(m.vertices)
    assert np.array_equal(m_efs.vertices[idx_uv_to_ef], m.vertices)

    # ----- Reverse Texture Lookup ----- #
    tex_img = np.copy(np.asarray(m.visual.material.image))
    H, W, _ = tex_img.shape
    dtype = np.float128  # Doesn't seem to help much to use float128 (but maybe it's not really 128 bit on my platform)

    # Face Vertices
    pu = (W - 1) * m.visual.uv[:, 0].astype(dtype)
    pv = (H - 1) * (1 - m.visual.uv[:, 1]).astype(dtype)
    puvs = np.stack([pu, pv], -1)
    a = puvs[m.faces[:, 0]].astype(dtype)
    b = puvs[m.faces[:, 1]].astype(dtype)
    c = puvs[m.faces[:, 2]].astype(dtype)

    PX, PY = np.meshgrid(np.arange(W), np.arange(H))
    p = np.stack([PX.ravel(), PY.ravel()], -1).astype(dtype)

    # idx = get_tris_fast(p=p, a=a, b=b, c=c)
    chunk_size = 1 << 15
    chunks = np.split(p, np.arange(chunk_size, p.shape[0], chunk_size), axis=0)
    idx_chunks = []
    for chunk in chunks:
        idx_chunks.append(get_tris_fast(p=chunk, a=a, b=b, c=c))
    idx = np.concatenate(idx_chunks, axis=0)

    p_val = p[idx >= 0]
    idx_val = idx[idx >= 0]
    u, v, w = bary_matched(p=p_val, a=a[idx_val], b=b[idx_val], c=c[idx_val])
    bari = np.stack([u, v, w], -1)

    faces = m.faces[idx_val]

    # Create z colormap for debug
    xyz_rec = xyz_from_faces_and_bary(vert=m.vertices, faces=faces, bari=bari)
    val = xyz_rec[:, 2]
    val = (val - val.min()) / (val.max() - val.min())
    assert val.max() == 1
    cols = plt.cm.viridis(val)[..., :3]

    cols_all = np.zeros([p.shape[0], cols.shape[-1]])
    cols_all[idx >= 0] = cols

    CC = np.reshape(cols_all, [PX.shape[0], PX.shape[1], cols_all.shape[-1]])
    CC_filled = uv_fill_holes(CC)
    tex_colormap = (255 * CC_filled).astype(np.uint8)

    os.makedirs(os.path.join(out_dir, "colormap"), exist_ok=False)
    shutil.copyfile(src=uv_mesh_path, dst=os.path.join(out_dir, "colormap", os.path.basename(uv_mesh_path)))
    shutil.copyfile(src=uv_mesh_path + ".mtl",
                    dst=os.path.join(out_dir, "colormap", os.path.basename(uv_mesh_path) + ".mtl"))

    diffuse_color_map_name = get_diffuse_color_map_file_name(uv_mesh_path)
    assert cv2.imwrite(os.path.join(out_dir, "colormap", diffuse_color_map_name), tex_colormap[..., ::-1])

    # Avoid misuse
    del val, cols, cols_all, CC, CC_filled, tex_colormap

    # Get faces in eigenfunction mesh
    faces_index_efs = idx_uv_to_ef[faces]

    xyz_rec_efs = xyz_from_faces_and_bary(m_efs.vertices, faces=faces_index_efs, bari=bari)
    assert np.array_equal(xyz_rec, xyz_rec_efs)
    assert faces_index_efs.ndim == 2 and faces_index_efs.shape[-1] == 3
    assert bari.shape == faces_index_efs.shape
    assert np.allclose(np.sum(bari, -1), 1)
    assert np.all(bari >= 0)

    print(f"Computing RGBs for {len(bari)} points")
    rgbs = pred_rgbs(m_efs, faces_index_efs, bari, config)

    assert rgbs.shape == bari.shape
    assert np.all(rgbs >= 0) and np.all(rgbs <= 1)
    print(f"{np.sum(np.all(rgbs == 0, axis=-1)).item()}/{len(rgbs)} pixel were rendered with invalid color RGB=(0,0,0)")

    cols_all = np.zeros([p.shape[0], 3])
    cols_all[idx >= 0] = rgbs

    CC = np.reshape(cols_all, [PX.shape[0], PX.shape[1], cols_all.shape[-1]])
    CC_filled = uv_fill_holes(CC)
    tex_baked = (255 * CC_filled).astype(np.uint8)

    os.makedirs(os.path.join(out_dir, "baked"), exist_ok=False)
    shutil.copyfile(src=uv_mesh_path, dst=os.path.join(out_dir, "baked", os.path.basename(uv_mesh_path)))
    shutil.copyfile(src=uv_mesh_path + ".mtl", dst=os.path.join(out_dir, "baked", os.path.basename(uv_mesh_path) + ".mtl"))

    assert cv2.imwrite(os.path.join(out_dir, "baked", diffuse_color_map_name), tex_baked[..., ::-1])


if __name__ == "__main__":
    args = parse_args()
    bake_texture(args.out_dir, args.uv_mesh, args.config_path)
