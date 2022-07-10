import numpy as np
import torch
import igl
import trimesh
import scipy as sp
import os
from functools import lru_cache
import gc

from cameras import undistort_pixels_meshroom_radial_k3, DistortionTypes
from utils import tensor_mem_size_in_bytes


@lru_cache(maxsize=8)
def is_pointcloud_file(path):
    return isinstance(trimesh.load(path), trimesh.PointCloud)


def load_pointcloud(path):
    """
    This function loads a point cloud and creates a local triangulation around the vertices and returns the
    local triangulated point cloud as a mesh.
    """
    pc = trimesh.load(path)
    vertices = np.array(pc.vertices)

    from potpourri3d import PointCloudLocalTriangulation  # Note: Custom potpourri3d!
    local_triangulation = PointCloudLocalTriangulation(vertices)
    faces = local_triangulation.get_local_triangulation()

    valid_face_idxs = np.all(faces >= 0, -1)
    faces = faces[valid_face_idxs]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, validate=False, process=False, maintain_order=True)
    assert np.array_equal(vertices, mesh.vertices) and np.array_equal(faces, mesh.faces)
    return mesh


def load_mesh(path):
    if is_pointcloud_file(path):
        return load_pointcloud(path)

    # Note: We load using libigl because trimesh does some unwanted preprocessing and vertex
    # reordering (even if process=False and maintain_order=True is set). Hence, we load it
    # using libigl and then convert the loaded mesh into a Trimesh object.
    v, f = igl.read_triangle_mesh(path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)
    # assert mesh.is_watertight and (v == mesh.vertices).all() and (f == mesh.faces).all()
    assert np.array_equal(v, mesh.vertices) and np.array_equal(f, mesh.faces)
    return mesh


def load_first_k_eigenfunctions(eigenfunctions_path,
                                k,
                                rescale_strategy="standard",
                                embed_strategy=None,
                                eigenvalues_path=None,
                                ts=128):
    all_eigenfunctions = np.load(eigenfunctions_path)
    if isinstance(k, list):
        eigenfunctions = all_eigenfunctions[:, np.array(k)]
    else:
        stored_k = all_eigenfunctions.shape[1]
        assert k <= stored_k
        eigenfunctions = all_eigenfunctions[:, :k]

    if eigenvalues_path is not None:
        all_eigenvalues = np.load(eigenvalues_path)
        if isinstance(k, list):
            eigenvalues = all_eigenvalues[:, np.array(k)]
        else:
            stored_k = all_eigenvalues.shape[0]
            assert k <= stored_k
            eigenvalues = all_eigenvalues[:k]

        if np.abs(eigenvalues[0]) < 1e-10 and eigenvalues[0] < 0:
            # The first eigenvalue might be negative and close to 0 due to numerical instability. Hence, we fix it.
            # The other eigenvalues are expected to be > 0.
            eigenvalues[0] *= -1

        assert np.all(eigenvalues > 0), f"Min value: {eigenvalues.min()}"

    if embed_strategy is not None:
        if embed_strategy == "gps":
            # GPS (Global Point Signature)
            assert eigenvalues is not None
            weights = np.sqrt(eigenvalues)
            weights /= weights[0]
            return eigenfunctions / weights
            # eigenfunctions = eigenfunctions / np.sqrt(eigenvalues)
        elif embed_strategy == "hks":
            # HKS (Heat Kernel Signature)
            assert eigenvalues is not None
            timesteps = np.logspace(-2, 0, num=ts)
            eigenfunctions = (eigenfunctions * eigenfunctions) @ np.exp(-eigenvalues[..., None] @ timesteps[None, ...])
        else:
            raise ValueError(f"Unknown embedding strategy {embed_strategy}")

    if rescale_strategy == "standard":
        # Rescale the eigenfunctions, s.t. they are in [-1;1].
        eigenfunctions = eigenfunctions / \
                         (np.max(eigenfunctions, axis=0, keepdims=True) - np.min(eigenfunctions, axis=0, keepdims=True))
    elif rescale_strategy == "one-norm":  # TODO(daniel): rename to euclidean-norm
        eigenfunctions = eigenfunctions / np.linalg.norm(eigenfunctions, ord=2, axis=-1, keepdims=True)
    elif rescale_strategy != "unscaled":
        raise RuntimeError(f"Unknown rescaling strategy: {rescale_strategy}")

    return torch.from_numpy(eigenfunctions).to(dtype=torch.float32)


def get_ray_mesh_intersector(mesh):
    try:
        import pyembree
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except ImportError:
        print("Warning: Could not find pyembree, the ray-mesh intersection will be significantly slower.")
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def compute_first_k_eigenfunctions(mesh, k, laplacian_type="cotan", skip_first_efunc=True, return_evalues=False):
    if laplacian_type == "cotan":
        # Compute cotan and mass matrix
        L = -igl.cotmatrix(mesh.vertices, mesh.faces)  # -L to make eigenvalues positive
        M = igl.massmatrix(mesh.vertices, mesh.faces, igl.MASSMATRIX_TYPE_VORONOI)
    elif laplacian_type == "robust":
        import robust_laplacian
        # Use Robust Laplacian from: Sharp and Crane "A Laplacian for Nonmanifold Triangle Meshes"
        L, M = robust_laplacian.mesh_laplacian(np.array(mesh.vertices), np.array(mesh.faces))
    elif laplacian_type == "pc_vert_robust":
        import robust_laplacian
        # Use vertices of mesh as point in a point cloud
        # Use Robust Laplacian from: Sharp and Crane "A Laplacian for Nonmanifold Triangle Meshes"
        L, M = robust_laplacian.point_cloud_laplacian(np.array(mesh.vertices))
    else:
        raise RuntimeError(f"Laplacian type {laplacian_type} not implemented.")

    # k + 1 because we will remove the first eigenfunction since it is always a constant
    # but we still want k eigenfunctions.
    try:
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k + 1, M, sigma=0, which="LM")
    except RuntimeError as e:
        if len(e.args) == 1 and e.args[0] == "Factor is exactly singular":
            print(
                "Stiffness matrix L is singular because L is most likely badly conditioned. Retrying with improved condition...")
            # https://stackoverflow.com/questions/18754324/improving-a-badly-conditioned-matrix
            c = 1e-10
            L = L + c * sp.sparse.eye(L.shape[0])
            # Retry
            eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k + 1, M, sigma=0, which="LM")

    # This must hold true otherwise we would divide by 0 later on!
    assert np.all(np.max(eigenfunctions, axis=0) != np.min(eigenfunctions, axis=0))

    # We remove the first eigenfunction since it is constant. This implies that for
    # any linear layer Wx+b (assuming W and b are scalars for now), we would get W+b
    # which is always a bias.
    if skip_first_efunc:
        eigenfunctions = eigenfunctions[:, 1:]
        eigenvalues = eigenvalues[1:]
    else:
        # Remove the +1 again
        eigenfunctions = eigenfunctions[:, :-1]
        eigenvalues = eigenvalues[:-1]

    if return_evalues:
        return eigenfunctions, eigenvalues

    return eigenfunctions


def create_ray_origins_and_directions(camCv2world, K, mask_1d, *, H, W, distortion_coeffs=None, distortion_type=None):
    # Let L be the number of pixels where the object is seen in the view
    L = mask_1d.sum()

    try:
        # This does not work for older PyTorch versions.
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    except TypeError:
        # Workaround for older versions. Simulate indexing='xy'
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        coord2d_x, coord2d_y = coord2d_x.T, coord2d_y.T

    coord2d = torch.cat([coord2d_x[..., None], coord2d_y[..., None]], dim=-1).reshape(-1, 2)  # N*M x 2
    selected_coord2d = coord2d[mask_1d]  # L x 2
    
    # If the views are distorted, remove the distortion from the 2D pixel coordinates
    if distortion_type is not None:
        assert distortion_coeffs is not None
        if distortion_type == DistortionTypes.MESHROOM_RADIAL_K3:
            selected_coord2d = undistort_pixels_meshroom_radial_k3(selected_coord2d.numpy(), K.numpy(), distortion_coeffs)
            selected_coord2d = torch.from_numpy(selected_coord2d).to(torch.float32)
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

    # Get the 3D world coordinates of the ray origins as well as the 3D unit direction vector

    # Origin of the rays of the current view (it is already in 3D world coordinates)
    ray_origins = camCv2world[:, 3].unsqueeze(0).expand(L, -1)  # L x 3

    # Transform 2d coordinates into homogeneous coordinates.
    selected_coord2d = torch.cat((selected_coord2d, torch.ones((L, 1))), dim=-1)  # L x 3
    # Calculate the ray direction: R (K^-1_{3x3} [u v 1]^T)
    ray_dirs = camCv2world[:3, :3].matmul(K[:3, :3].inverse().matmul(selected_coord2d.T)).T  # L x 3
    unit_ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    assert unit_ray_dirs.dtype == torch.float32

    return ray_origins, unit_ray_dirs


def ray_mesh_intersect(ray_mesh_intersector, mesh, ray_origins, ray_directions, return_depth=False, camCv2world=None):
    # Compute the intersection points between the mesh and the rays

    # Note: It might happen that M <= N where M is the number of returned hits
    intersect_locs, hit_ray_idxs, face_idxs = \
        ray_mesh_intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

    # Next, we need to determine the barycentric coordinates of the hit points.

    vertex_idxs_of_hit_faces = torch.from_numpy(mesh.faces[face_idxs]).reshape(-1)  # M*3
    hit_triangles = mesh.vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1, 3)  # M x 3

    barycentric_coords = trimesh.triangles.points_to_barycentric(hit_triangles, intersect_locs, method='cramer')  # M x 3

    if return_depth:
        assert camCv2world is not None
        camCv2world = camCv2world.cpu().numpy()
        camCv2world = np.concatenate([camCv2world, np.array([[0., 0, 0, 1]], dtype=camCv2world.dtype)], 0)

        vertices_world = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], -1)  # V, 4

        camWorld2Cv = np.linalg.inv(camCv2world)
        vertices_cam = np.dot(vertices_world, camWorld2Cv.T)
        z_vals = vertices_cam[:, 2][vertex_idxs_of_hit_faces]
        assert np.all(z_vals > 0)

        assert z_vals.shape == barycentric_coords.shape
        assert np.allclose(np.sum(barycentric_coords, -1), 1)

        hit_depth = np.sum(z_vals * barycentric_coords, -1)
        hit_depth = torch.from_numpy(hit_depth)

    barycentric_coords = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)  # M x 3

    hit_ray_idxs = torch.from_numpy(hit_ray_idxs)
    face_idxs = torch.from_numpy(face_idxs).to(dtype=torch.int64)

    if return_depth:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth
    return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs


def ray_mesh_intersect_batched(ray_mesh_intersector, mesh, ray_origins, ray_directions):
    batch_size = 1 << 18
    num_rays = ray_origins.shape[0]
    idxs = np.arange(0, num_rays)
    batch_idxs = np.split(idxs, np.arange(batch_size, num_rays, batch_size), axis=0)

    total_vertex_idxs_of_hit_faces = []
    total_barycentric_coords = []
    total_hit_ray_idxs = []
    total_face_idxs = []

    total_hits = 0
    hit_ray_idx_offset = 0
    for cur_idxs in batch_idxs:
        cur_ray_origins = ray_origins[cur_idxs]
        cur_ray_dirs = ray_directions[cur_idxs]

        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   cur_ray_origins,
                                                                                                   cur_ray_dirs)

        # Correct the hit_ray_idxs
        hit_ray_idxs += hit_ray_idx_offset

        num_hits = vertex_idxs_of_hit_faces.shape[0]

        # Append results to output
        if num_hits > 0:
            total_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces)
            total_barycentric_coords.append(barycentric_coords)
            total_hit_ray_idxs.append(hit_ray_idxs)
            total_face_idxs.append(face_idxs)

        hit_ray_idx_offset += cur_idxs.shape[0]
        total_hits += num_hits

    # Concatenate results
    out_vertex_idxs_of_hit_faces = torch.zeros((total_hits, 3), dtype=torch.int64)
    out_barycentric_coords = torch.zeros((total_hits, 3), dtype=torch.float32)
    out_hit_ray_idxs = torch.zeros(total_hits, dtype=torch.int64)
    out_face_idxs = torch.zeros(total_hits, dtype=torch.int64)

    offset = 0
    for i in range(len(total_vertex_idxs_of_hit_faces)):
        hits_of_batch = total_vertex_idxs_of_hit_faces[i].shape[0]
        low = offset
        high = low + hits_of_batch

        out_vertex_idxs_of_hit_faces[low:high] = total_vertex_idxs_of_hit_faces[i]
        out_barycentric_coords[low:high] = total_barycentric_coords[i]
        out_hit_ray_idxs[low:high] = total_hit_ray_idxs[i]
        out_face_idxs[low:high] = total_face_idxs[i]

        offset = high

    return out_vertex_idxs_of_hit_faces, out_barycentric_coords, out_hit_ray_idxs, out_face_idxs


def get_k_eigenfunc_vec_vals(E, vertex_idxs_of_hit_faces, barycentric_coords):
    B = vertex_idxs_of_hit_faces.size()[0]

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1)  # B*3

    # Get for each vertex of the hit face their corresponding eigenfunction vector values
    eigenfuncs_triangle = E[vertex_idxs_of_hit_faces]  # B*3 x k
    eigenfuncs_triangle = eigenfuncs_triangle.reshape(B, 3, -1)  # B x 3 x k

    # Using the barycentric coordinates, we compute the eigenfunction vector values of the point
    eigenfunc_vec_vals = torch.bmm(barycentric_coords.unsqueeze(1), eigenfuncs_triangle)  # B x 1 x k
    return eigenfunc_vec_vals.squeeze(1)  # B x k


def get_k_eigenfunc_vec_vals_batched(E, vertex_idxs_of_hit_faces, barycentric_coords):
    batch_size = 1 << 18
    B = vertex_idxs_of_hit_faces.size()[0]

    eigenfunc_vec_vals = torch.zeros((B, E.shape[1]), dtype=torch.float32)
    for i in range((B + batch_size - 1) // batch_size):
        low = i * batch_size
        high = min(B, (i + 1) * batch_size)

        eigenfunc_vec_vals[low:high] = get_k_eigenfunc_vec_vals(E, vertex_idxs_of_hit_faces[low:high],
                                                                barycentric_coords[low:high])

    return eigenfunc_vec_vals


def ray_tracing(ray_mesh_intersector,
                mesh,
                eigenfunctions,
                camCv2world,
                K,
                obj_mask_1d=None,
                *,
                H,
                W,
                batched=True,
                distortion_coeffs=None, 
                distortion_type=None):
    if obj_mask_1d is None:
        mask = torch.tensor([True]).expand(H * W)
    else:
        mask = obj_mask_1d

    ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                   K, 
                                                                   mask, 
                                                                   H=H, 
                                                                   W=W, 
                                                                   distortion_coeffs=distortion_coeffs,
                                                                   distortion_type=distortion_type)
    if batched:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect_batched(
            ray_mesh_intersector,
            mesh,
            ray_origins,
            unit_ray_dirs)
        first_k_eigenfunctions = get_k_eigenfunc_vec_vals_batched(eigenfunctions,
                                                                  vertex_idxs_of_hit_faces,
                                                                  barycentric_coords)
        return first_k_eigenfunctions, hit_ray_idxs, unit_ray_dirs[hit_ray_idxs], face_idxs

    vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(
        ray_mesh_intersector,
        mesh,
        ray_origins,
        unit_ray_dirs)
    first_k_eigenfunctions = get_k_eigenfunc_vec_vals(eigenfunctions,
                                                      vertex_idxs_of_hit_faces,
                                                      barycentric_coords)
    return first_k_eigenfunctions, hit_ray_idxs, unit_ray_dirs[hit_ray_idxs], face_idxs


def ray_tracing_xyz(ray_mesh_intersector,
                    mesh,
                    vertices,
                    camCv2world,
                    K,
                    obj_mask_1d=None,
                    *,
                    H,
                    W,
                    batched=True,
                    distortion_coeffs=None, 
                    distortion_type=None):
    if obj_mask_1d is None:
        mask = torch.tensor([True]).expand(H * W)
    else:
        mask = obj_mask_1d
    ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                   K, 
                                                                   mask, 
                                                                   H=H, 
                                                                   W=W, 
                                                                   distortion_coeffs=distortion_coeffs,
                                                                   distortion_type=distortion_type)
    if batched:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect_batched(
            ray_mesh_intersector,
            mesh,
            ray_origins,
            unit_ray_dirs)
    else:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   ray_origins,
                                                                                                   unit_ray_dirs)

    # Calculate the xyz hit points using the barycentric coordinates
    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1)  # M*3
    face_vertices = vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3
    hit_points_xyz = torch.einsum('bij,bi->bj', face_vertices, barycentric_coords)  # M x 3

    return hit_points_xyz, hit_ray_idxs, unit_ray_dirs[hit_ray_idxs], face_idxs


class MeshViewPreProcessor:
    def __init__(self, path_to_mesh, out_directory):
        self.out_dir = out_directory
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)

        self.cache_vertex_idxs_of_hit_faces = []
        self.cache_barycentric_coords = []
        self.cache_expected_rgbs = []
        self.cache_unit_ray_dirs = []
        self.cache_face_idxs = []

    def _ray_mesh_intersect(self, ray_origins, ray_directions, return_depth=False, camCv2world=None):
        return ray_mesh_intersect(self.ray_mesh_intersector, 
                                  self.mesh, 
                                  ray_origins, 
                                  ray_directions, 
                                  return_depth=return_depth, 
                                  camCv2world=camCv2world)

    def cache_single_view(self, camCv2world, K, mask, img, depth_check=None, distortion_coeffs=None, distortion_type=None):
        H, W = mask.shape

        mask = mask.reshape(-1)  # H*W
        img = img.reshape(H * W, -1)  # H*W x 3

        # Let L be the number of pixels where the object is seen in the view

        # Get the expected RGB value of the intersection points with the mesh
        expected_rgbs = img[mask]  # L x 3

        # Get the ray origins and unit directions
        ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                       K, 
                                                                       mask, 
                                                                       H=H, 
                                                                       W=W, 
                                                                       distortion_coeffs=distortion_coeffs, 
                                                                       distortion_type=distortion_type)

        # Then, we can compute the ray-mesh-intersections
        if depth_check is not None:
            vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth = \
                self._ray_mesh_intersect(ray_origins, unit_ray_dirs, return_depth=True, camCv2world=camCv2world)
        else:
            vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = self._ray_mesh_intersect(ray_origins, unit_ray_dirs)

        # If depth is supplied we do a depth check
        if depth_check is not None:
            assert depth_check.shape == (H, W)
            depth_check = depth_check.reshape(-1)  # H*W
            depth_masked = depth_check[mask]  # L

            hit_depth = hit_depth.cpu().numpy()
            hit_depth_check = depth_masked[hit_ray_idxs]
            outlier_thres = np.mean(hit_depth_check)*1e-2

            depth_error = np.abs(hit_depth-hit_depth_check)
            inlier_mask = depth_error < outlier_thres
            if np.sum(inlier_mask) < inlier_mask.size:
                print(f"DEPTH CHECK: Removed {inlier_mask.size-np.sum(inlier_mask)} = {100-100*np.mean(inlier_mask):6.3f} % of points")
                print(f"DEPTH CHECK: Mean depth error remaining {np.mean(depth_error[inlier_mask]):7.4f}")

            # Select
            vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces[inlier_mask]
            barycentric_coords = barycentric_coords[inlier_mask]
            hit_ray_idxs = hit_ray_idxs[inlier_mask]
            face_idxs = face_idxs[inlier_mask]
            del hit_depth

        # Choose the correct GTs and viewing directions for the hits.
        num_hits = hit_ray_idxs.size()[0]
        expected_rgbs = expected_rgbs[hit_ray_idxs]
        unit_ray_dirs = unit_ray_dirs[hit_ray_idxs]

        # Some clean up to free memory
        del ray_origins, hit_ray_idxs, mask, img
        gc.collect()  # Force garbage collection

        # Cast the indices down to int32 to save memory. Usually indices have to be int64, however, we assume that
        # the indices from 0 to 2^31-1 are sufficient. Therefore, we can savely cast down
        assert torch.all(face_idxs <= (2<<31)-1)
        face_idxs = face_idxs.to(torch.int32)
        assert torch.all(vertex_idxs_of_hit_faces <= (2<<31)-1)
        vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.to(torch.int32)
        barycentric_coords = barycentric_coords.to(torch.float32)
        expected_rgbs = expected_rgbs.to(torch.float32)
        unit_ray_dirs = unit_ray_dirs.to(torch.float32)

        # And finally, we store the results in the cache
        for idx in range(num_hits):
            self.cache_face_idxs.append(face_idxs[idx])
            self.cache_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces[idx])
            self.cache_barycentric_coords.append(barycentric_coords[idx])
            self.cache_expected_rgbs.append(expected_rgbs[idx])
            self.cache_unit_ray_dirs.append(unit_ray_dirs[idx])

    def write_to_disk(self):
        print("Starting to write to disk...")

        # Write the cached eigenfuncs and cached expected RGBs to disk
        os.makedirs(self.out_dir, exist_ok=True)

        # Stack the results, write to disk, and then free up memory

        self.cache_face_idxs = torch.stack(self.cache_face_idxs)
        print(
            f"Face Idxs: dim={self.cache_face_idxs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_face_idxs)}B, dtype={self.cache_face_idxs.dtype}")
        np.save(os.path.join(self.out_dir, "face_idxs.npy"), self.cache_face_idxs, allow_pickle=False)
        del self.cache_face_idxs
        gc.collect()  # Force garbage collection

        self.cache_vertex_idxs_of_hit_faces = torch.stack(self.cache_vertex_idxs_of_hit_faces)
        print(
            f"Vertex Idxs of Hit Faces: dim={self.cache_vertex_idxs_of_hit_faces.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_vertex_idxs_of_hit_faces)}B, dtype={self.cache_vertex_idxs_of_hit_faces.dtype}")
        np.save(os.path.join(self.out_dir, "vids_of_hit_faces.npy"), self.cache_vertex_idxs_of_hit_faces,
                allow_pickle=False)
        del self.cache_vertex_idxs_of_hit_faces
        gc.collect()  # Force garbage collection

        self.cache_barycentric_coords = torch.stack(self.cache_barycentric_coords)
        print(
            f"Barycentric Coords: dim={self.cache_barycentric_coords.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_barycentric_coords)}B, dtype={self.cache_barycentric_coords.dtype}")
        np.save(os.path.join(self.out_dir, "barycentric_coords.npy"), self.cache_barycentric_coords, allow_pickle=False)
        del self.cache_barycentric_coords
        gc.collect()  # Force garbage collection

        self.cache_expected_rgbs = torch.stack(self.cache_expected_rgbs)
        print(
            f"Expected RGBs: dim={self.cache_expected_rgbs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_expected_rgbs)}B, dtype={self.cache_expected_rgbs.dtype}")
        np.save(os.path.join(self.out_dir, "expected_rgbs.npy"), self.cache_expected_rgbs, allow_pickle=False)
        del self.cache_expected_rgbs
        gc.collect()  # Force garbage collection

        self.cache_unit_ray_dirs = torch.stack(self.cache_unit_ray_dirs)
        print(
            f"Unit Ray Dirs: dim={self.cache_unit_ray_dirs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_unit_ray_dirs)}B, dtype={self.cache_unit_ray_dirs.dtype}")
        np.save(os.path.join(self.out_dir, "unit_ray_dirs.npy"), self.cache_unit_ray_dirs, allow_pickle=False)
        del self.cache_unit_ray_dirs
        gc.collect()  # Force garbage collection


class EigenfuncsProcessor:
    def __init__(self, path_to_mesh, k, laplacian_type="cotan", skip_first_efunc=True):
        self.mesh = load_mesh(path_to_mesh)
        self.k = k
        self.laplacian_type = laplacian_type
        efuncs, evalues = compute_first_k_eigenfunctions(self.mesh, self.k, self.laplacian_type,
                                                         skip_first_efunc=skip_first_efunc, return_evalues=True)
        self.E = torch.from_numpy(efuncs).to(dtype=torch.float32)
        self.evalues = torch.from_numpy(evalues).to(dtype=torch.float32)

    def get_eigenfunctions(self):
        return self.E

    def get_eigenvalues(self):
        return self.evalues


def get_remapped_efuncs_with_fm_gt(k, target_efuncs_path, source_efuncs_path, source_mesh, eigenvalues_path):
    E_target = load_first_k_eigenfunctions(target_efuncs_path, k, rescale_strategy="unscaled",
                                           eigenvalues_path=eigenvalues_path).numpy()

    E_source = load_first_k_eigenfunctions(source_efuncs_path, k, rescale_strategy="unscaled",
                                           eigenvalues_path=eigenvalues_path).numpy()
    M_source = igl.massmatrix(source_mesh.vertices, source_mesh.faces, igl.MASSMATRIX_TYPE_VORONOI).astype(np.float32)

    # Functional Map matrix GT
    C_source_target = E_source.T @ M_source @ E_target  # k_source x k_target

    mapped_source_efuncs = E_source @ C_source_target  # N_source x k_target
    mapped_source_efuncs /= np.max(mapped_source_efuncs, axis=0, keepdims=True) - np.min(mapped_source_efuncs, axis=0,
                                                                                         keepdims=True)

    return torch.from_numpy(mapped_source_efuncs)
