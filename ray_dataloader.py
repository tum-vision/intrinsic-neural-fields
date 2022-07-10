import torch

from mesh import get_k_eigenfunc_vec_vals, load_first_k_eigenfunctions
from dataset import load_preprocessed_data


def create_ray_dataloader(preproc_data_path,
                          eigenfunctions_path,
                          k,
                          feature_strategy,
                          mesh,
                          rescale_strategy,
                          eigenvalues_path,
                          embed_strategy,
                          batch_size,
                          shuffle,
                          drop_last,
                          device="cuda"):
    # LOAD DATA

    # Get input features
    if feature_strategy == "efuncs":
        features = load_first_k_eigenfunctions(eigenfunctions_path,
                                               k,
                                               rescale_strategy=rescale_strategy,
                                               embed_strategy=embed_strategy,
                                               eigenvalues_path=eigenvalues_path)
    elif feature_strategy in ("ff", "rff", "xyz"):
        assert mesh is not None
        features = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown input feature strategy: {feature_strategy}")

    # Get ray-mesh intersection information
    data = load_preprocessed_data(preproc_data_path)
    vertex_idxs_of_hit_faces = data["vertex_idxs_of_hit_faces"]
    barycentric_coords = data["barycentric_coords"]
    expected_rgbs = data["expected_rgbs"]

    # Get view dependence information
    unit_ray_dirs = data.get("unit_ray_dirs")
    face_idxs = data.get("face_idxs")

    return RayDataLoader(features,
                         feature_strategy,
                         vertex_idxs_of_hit_faces,
                         barycentric_coords,
                         expected_rgbs,
                         unit_ray_dirs,
                         face_idxs,
                         batch_size,
                         shuffle,
                         drop_last,
                         device=device)


class RayDataLoader:
    def __init__(self,
                 features,
                 feature_strategy,
                 vertex_idxs_of_hit_faces,
                 barycentric_coords,
                 expected_rgbs,
                 unit_ray_dirs,
                 face_idxs,
                 batch_size,
                 shuffle,
                 drop_last,
                 device="cuda"):
        self.device = device

        # INITIALIZE DATA AND MOVE TO DEVICE
        self.features = features.to(self.device)
        self.feature_strategy = feature_strategy
        self.vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.to(self.device)
        self.barycentric_coords = barycentric_coords.to(self.device)
        self.expected_rgbs = expected_rgbs.to(self.device)
        self.unit_ray_dirs = unit_ray_dirs
        self.face_idxs = face_idxs
        if self.unit_ray_dirs is not None:
            assert self.face_idxs is not None
            self.unit_ray_dirs = self.unit_ray_dirs.to(self.device)
            self.face_idxs = self.face_idxs.to(self.device)

        # DATALOADING SPECIFICS

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.B = batch_size
        self.N = self.vertex_idxs_of_hit_faces.shape[0]
        if self.drop_last:
            self.num_batches = self.N // self.B
        else:
            self.num_batches = (self.N + self.B - 1) // self.B

        self.i = 0
        self.idxs = torch.arange(self.N, device=self.device)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(self.N, device=self.device)
        self.i = 0
        return self

    def _get_next_batch_idxs(self):
        low = self.i * self.B
        high = min((self.i + 1) * self.B, self.N)
        self.i += 1
        return self.idxs[low:high]

    def __next__(self):
        if self.i >= self.num_batches:
            raise StopIteration

        batch_idxs = self._get_next_batch_idxs()

        batch = {
            "expected_rgbs": self.expected_rgbs[batch_idxs],
        }

        vertex_idxs_of_hit_faces_local = self.vertex_idxs_of_hit_faces[batch_idxs]  # B x 3
        barycentric_coords = self.barycentric_coords[batch_idxs]  # B x 3

        if self.feature_strategy == "efuncs":
            efuncs_local = get_k_eigenfunc_vec_vals(self.features,
                                                    vertex_idxs_of_hit_faces_local,
                                                    barycentric_coords)
            assert efuncs_local.dtype == torch.float32
            batch["eigenfunctions"] = efuncs_local
        elif self.feature_strategy in ("ff", "rff", "xyz"):
            features_local = self.features[vertex_idxs_of_hit_faces_local]  # B x 3 x 3
            batch["xyz"] = torch.bmm(barycentric_coords.unsqueeze(1), features_local).squeeze(1)  # B x 3
        else:
            raise ValueError(f"Unknown input feature strategy: {self.feature_strategy}")

        if self.unit_ray_dirs is not None:
            assert self.face_idxs is not None
            batch["unit_ray_dirs"] = self.unit_ray_dirs[batch_idxs]
            batch["hit_face_idxs"] = self.face_idxs[batch_idxs]

        return batch


# TESTING
if __name__ == "__main__":
    device = "cuda"

    vertex_idxs_of_hit_faces = torch.tensor([[0, 1, 2],
                                             [1, 2, 3],
                                             [7, 8, 9],
                                             [5, 6, 7],
                                             [3, 4, 5]
                                             ])
    barycentric = torch.tensor([[1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0]], dtype=torch.float32)
    expected_rgbs = torch.ones((5, 3), dtype=torch.float32)
    batch_size = 2

    # Test Intrinsic Loading
    k = 5
    efuncs = torch.rand((10, k))
    intrinsic_dataloader = RayDataLoader(efuncs,
                                         "efuncs",
                                         vertex_idxs_of_hit_faces,
                                         barycentric,
                                         expected_rgbs,
                                         None,
                                         None,
                                         batch_size,
                                         False,
                                         True,
                                         device=device)

    total_elements = 0
    for batch in intrinsic_dataloader:
        assert (batch_size, k) == batch["eigenfunctions"].shape
        total_elements += batch["eigenfunctions"].shape[0]
    assert total_elements == ((barycentric.shape[0] // batch_size) * batch_size)

    # Test Extrinsic Loading
    vertices = torch.rand((10, 3))
    extrinsic_dataloader = RayDataLoader(vertices,
                                         "xyz",
                                         vertex_idxs_of_hit_faces,
                                         barycentric,
                                         expected_rgbs,
                                         None,
                                         None,
                                         batch_size,
                                         False,
                                         True,
                                         device=device)

    total_elements = 0
    for batch in extrinsic_dataloader:
        assert (batch_size, 3) == batch["xyz"].shape
        total_elements += batch["xyz"].shape[0]
    assert total_elements == ((barycentric.shape[0] // batch_size) * batch_size)
