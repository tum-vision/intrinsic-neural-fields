import torch
import torch.nn as nn
import torch.nn.functional as F

from neutex.neutex import make_neutex_train_wrapper_default
from layers import FourierFeatEnc, RandomFourierFeatEnc, LinearWithConcatAndActivation, Sine


RGB_COLOR_DIM = 3


class TextureField(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 skip_layer_idx,
                 input_feature_embed=None,
                 embed_dim=None,
                 embed_include_input=True,
                 embed_std=1.,
                 return_rgb=True,
                 out_dim=RGB_COLOR_DIM,
                 batchnorm=False,
                 activation=nn.ReLU):
        super(TextureField, self).__init__()
        assert num_layers > 2 and 0 < skip_layer_idx and skip_layer_idx < num_layers-1

        self.skip_layer_idx = skip_layer_idx

        layers = []

        self.input_feature_embed = input_feature_embed
        if self.input_feature_embed == "ff":
            self.embedding = FourierFeatEnc(embed_dim, include_input=embed_include_input)
            in_dim = 3 * embed_dim * 2 + (3 if embed_include_input else 0)
        elif self.input_feature_embed == "rff":
            self.embedding = RandomFourierFeatEnc(embed_dim, std=embed_std, include_input=embed_include_input)
            in_dim = embed_dim * 2 + (3 if embed_include_input else 0)
        else:
            self.embedding = None
        
        # Input layer
        if batchnorm:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim), 
                    activation(),
                    nn.BatchNorm1d(hidden_dim)
                )
            )
        else:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim), 
                    activation()
                )
            )

        # Hidden layers with an input skip connection
        for i in range(1, num_layers - 1):
            if i == skip_layer_idx:
                # At this layer, we inject the input again
                layers.append(
                    LinearWithConcatAndActivation(hidden_dim, 
                                                  in_dim, 
                                                  hidden_dim, 
                                                  batchnorm=batchnorm, 
                                                  activation=activation)
                )
            else:
                if batchnorm:
                    layers.append(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            activation(),
                            nn.BatchNorm1d(hidden_dim)
                        )
                    )
                else:
                    layers.append(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            activation()
                        )
                    )

        # Output layer
        layers.append(
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid() if return_rgb else nn.ReLU()
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        if self.input_feature_embed == "ff" or self.input_feature_embed == "rff":
            features = self.embedding(batch["xyz"])
        elif self.input_feature_embed == "xyz":
            features = batch["xyz"]
        else:
            features = batch["eigenfunctions"]

        res = features
        for i in range(len(self.layers)):
            if i == self.skip_layer_idx:
                res = self.layers[i](res, features)
            else:
                res = self.layers[i](res)
        return res


def calculate_angle_between_vectors(a, b):
    # https://discuss.pytorch.org/t/efficient-way-to-calculate-angles-of-normals-between-to-tensors/22471
    # assuming a and are of shape N x 3
    assert a.size() == b.size() and len(a.size()) == 2 and a.size()[1] == 3
    cos_theta = F.cosine_similarity(a, b, dim=-1)
    return torch.acos(cos_theta)


class TextureFieldWithViewDependency(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hidden_dim,
                 skip_layer_idx,
                 bottleneck_vec_dim,
                 in_dim_view_dir,
                 include_view_dir,
                 view_dir_embedding_size,
                 directional_hidden_dim,
                 input_feature_embed=None,
                 embed_dim=None,
                 embed_include_input=True,
                 embed_std=1.,
                 face_normals=None,
                 view_dir_strategy="intrinsic",
                 batchnorm=False,
                 activation=nn.ReLU):
        super(TextureFieldWithViewDependency, self).__init__()

        self.view_dir_strategy = view_dir_strategy
        if face_normals is not None:
            self.register_buffer("face_normals", face_normals, persistent=False)

        self.spatial_mlp = TextureField(num_layers,
                                        in_dim,
                                        hidden_dim,
                                        skip_layer_idx,
                                        input_feature_embed=input_feature_embed,
                                        embed_dim=embed_dim,
                                        embed_include_input=embed_include_input,
                                        embed_std=embed_std,
                                        return_rgb=False,
                                        out_dim=bottleneck_vec_dim,
                                        batchnorm=batchnorm,
                                        activation=activation)

        self.embedding = FourierFeatEnc(view_dir_embedding_size,
                                        include_input=include_view_dir,
                                        use_logspace=True)
        embedding_size = in_dim_view_dir * view_dir_embedding_size * 2
        if include_view_dir:
            embedding_size += in_dim_view_dir

        self.directional_mlp = nn.Sequential(
            nn.Linear(bottleneck_vec_dim + embedding_size, directional_hidden_dim),
            activation(),
            nn.Linear(directional_hidden_dim, RGB_COLOR_DIM),
            nn.Sigmoid()
        )

    def _get_embedded_view_dir(self, batch):
        if self.view_dir_strategy == "intrinsic":
            # Select the normal of the face on which the hit point resides.
            hit_face_normals = self.face_normals[batch["hit_face_idxs"]]
            # Calculate the angle between the normal of the hit face and the viewing direction
            # Note: We must return the viewing direction around so that it also points away from the surface.
            angles = calculate_angle_between_vectors(-batch["unit_ray_dirs"], hit_face_normals)
            return self.embedding(angles.unsqueeze(-1))
        elif self.view_dir_strategy == "extrinsic":
            return self.embedding(batch["unit_ray_dirs"])
        else:
            raise RuntimeError("Unknown viewing direction strategy.")

    def forward(self, batch):
        bottleneck_vector = self.spatial_mlp(batch)
        view_dir = self._get_embedded_view_dir(batch)
        return self.directional_mlp(torch.cat((bottleneck_vector, view_dir), dim=-1))


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def make_model(model_config, mesh=None):
    view_dependence_config = model_config.get("view_dependence")
    feature_strategy = model_config.get("feature_strategy", "efuncs")

    if model_config.get("type") == "neutex":
        assert "pretrained_path" in model_config
        return make_neutex_train_wrapper_default(model_config)

    if feature_strategy == "xyz":
        in_dim = 3
    elif hasattr(model_config, "hks_timesteps"):
        in_dim = model_config["hks_timesteps"]
    elif isinstance(model_config["k"], int):
        in_dim = model_config["k"]
    else:
        assert isinstance(model_config["k"], list)
        in_dim = len(model_config["k"])

    activation_fn = model_config.get("activation", "relu")
    if activation_fn == "relu":
        activation = nn.ReLU
    elif activation_fn == "sine":
        activation = Sine
    else:
        raise NotImplementedError(f"Activation function {activation_fn} not yet implemented.")

    if view_dependence_config is None:        
        model = TextureField(model_config["num_layers"],
                             in_dim,
                             model_config["mlp_hidden_dim"],
                             model_config["skip_layer_idx"],
                             input_feature_embed=feature_strategy,
                             embed_dim=model_config.get("k"),
                             embed_include_input=model_config.get("embed_include_input", True),
                             embed_std=model_config.get("embed_std", 1.),
                             batchnorm=model_config.get("batchnorm", False), 
                             activation=activation)
    else:
        assert mesh is not None
        face_normals = torch.from_numpy(mesh.face_normals.copy()).to(dtype=torch.float32)
        model = TextureFieldWithViewDependency(model_config["num_layers"],
                                               in_dim,
                                               model_config["mlp_hidden_dim"],
                                               model_config["skip_layer_idx"],
                                               view_dependence_config["bottleneck_vec_dim"],
                                               view_dependence_config["in_dim_view_dir"],
                                               view_dependence_config["include_view_dir"],
                                               view_dependence_config["embed_size"],
                                               view_dependence_config["directional_hidden_dim"],
                                               input_feature_embed=feature_strategy,
                                               embed_dim=model_config.get("k"),
                                               embed_include_input=model_config.get("embed_include_input", True),
                                               embed_std=model_config.get("embed_std", 1.),
                                               face_normals=face_normals,
                                               view_dir_strategy=view_dependence_config["strategy"],
                                               batchnorm=model_config.get("batchnorm", False),
                                               activation=activation)

    model.apply(init_weights)
    return model
